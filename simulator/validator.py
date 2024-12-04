import asyncio
import traceback
from typing import List

import aiohttp
import bittensor as bt
from bittensor.btlogging import logging as logger
from tenacity import RetryError

import dojo
from commons.dataset.synthetic import SyntheticAPI
from commons.orm import ORM
from commons.utils import get_epoch_time, get_new_uuid, set_expire_time, ttl_get_block
from dojo.protocol import (
    DendriteQueryResponse,
    FeedbackRequest,
    MultiScoreCriteria,
    TaskType,
)
from neurons.validator import Validator


class ValidatorSim(Validator):
    def __init__(self):
        super().__init__()
        logger.info("Starting Validator Simulator")
        self._last_block = None
        self._block_check_attempts = 0
        self.MAX_BLOCK_CHECK_ATTEMPTS = 3

    async def _try_reconnect_subtensor(self):
        """Attempt to reconnect to the subtensor network"""
        try:
            logger.info("Attempting to reconnect to subtensor...")
            self.subtensor = bt.subtensor(self.subtensor.config)
            self._block_check_attempts = 0
            return True
        except Exception as e:
            logger.error(f"Failed to reconnect to subtensor: {e}")
            return False

    @property
    def block(self):
        try:
            self._last_block = ttl_get_block(self.subtensor)
            self._block_check_attempts = 0
            return self._last_block
        except BrokenPipeError:
            self._block_check_attempts += 1
            if self._block_check_attempts >= self.MAX_BLOCK_CHECK_ATTEMPTS:
                logger.error("Multiple failed attempts to get block number, attempting reconnection")
                if asyncio.get_event_loop().run_until_complete(self._try_reconnect_subtensor()):
                    return self.block

            return self._last_block if self._last_block is not None else 0
        except Exception as e:
            logger.error(f"Error getting block number: {e}")
            return self._last_block if self._last_block is not None else 0

    def check_registered(self):
        new_subtensor = bt.subtensor(self.subtensor.config)
        if not new_subtensor.is_hotkey_registered(
                netuid=self.config.netuid,
                hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            logger.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli s register` before trying again"
            )
            exit()

    async def send_request(
            self,
            synapse: FeedbackRequest | None = None,
            external_user: bool = False,
    ):
        start = get_epoch_time()
        # typically the request may come from an external source however,
        # initially will seed it with some data for miners to get started
        if len(self._active_miner_uids) == 0:
            logger.info("No active miners to send request to... skipping")
            return

        request_id = get_new_uuid()
        # sel_miner_uids = await self.get_miner_uids(external_user, request_id)
        sel_miner_uids = sorted(list(self._active_miner_uids))

        axons = [
            self.metagraph.axons[uid]
            for uid in sel_miner_uids
            if self.metagraph.axons[uid].hotkey.casefold()
               != self.wallet.hotkey.ss58_address.casefold()
        ]
        if not len(axons):
            logger.warning("🤷 No axons to query ... skipping")
            return

        obfuscated_model_to_model = {}

        if synapse is None:
            try:
                data = await SyntheticAPI.get_qa()
            except RetryError as e:
                logger.error(
                    f"Exhausted all retry attempts for synthetic data generation: {e}"
                )
                return
            except ValueError as e:
                logger.error(f"Invalid response from synthetic data API: {e}")
                return
            except aiohttp.ClientError as e:
                logger.error(f"Network error when calling synthetic data API: {e}")
                return
            except Exception as e:
                logger.error(f"Unexpected error during synthetic data generation: {e}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                return

            if not data:
                logger.error("No data returned from synthetic data API")
                return

            obfuscated_model_to_model = self.obfuscate_model_names(data.responses)
            expire_at = set_expire_time(dojo.TASK_DEADLINE)
            synapse = FeedbackRequest(
                request_id=request_id,
                task_type=str(TaskType.CODE_GENERATION),
                criteria_types=[
                    MultiScoreCriteria(
                        options=list(obfuscated_model_to_model.keys()),
                        min=1.0,
                        max=100.0,
                    ),
                ],
                prompt=data.prompt,
                completion_responses=data.responses,
                expire_at=expire_at,
                ground_truth=data.ground_truth  # Added ground truth!!!!!
            )
        elif external_user:
            obfuscated_model_to_model = self.obfuscate_model_names(
                synapse.completion_responses
            )

        logger.info(
            f"⬆️ Sending feedback request for request id: {synapse.request_id}, miners uids:{sel_miner_uids} with expire_at: {synapse.expire_at}"
        )

        miner_responses: List[FeedbackRequest] = await self._send_shuffled_requests(
            self.dendrite, axons, synapse
        )

        valid_miner_responses: List[FeedbackRequest] = []
        try:
            for miner_response in miner_responses:
                miner_hotkey = (
                    miner_response.axon.hotkey if miner_response.axon else "??"
                )
                # map obfuscated model names back to the original model names
                real_model_ids = []

                for i, completion in enumerate(miner_response.completion_responses):
                    found_model_id = obfuscated_model_to_model.get(
                        completion.model, None
                    )
                    real_model_ids.append(found_model_id)
                    if found_model_id:
                        miner_response.completion_responses[i].model = found_model_id
                        synapse.completion_responses[i].model = found_model_id

                if any(c is None for c in real_model_ids):
                    logger.warning("Failed to map obfuscated model to original model")
                    continue

                if miner_response.dojo_task_id is None:
                    logger.debug(f"Miner {miner_hotkey} must provide the dojo task id")
                    continue

                # update the miner response with the real model ids
                valid_miner_responses.append(miner_response)
        except Exception as e:
            logger.error(f"Failed to map obfuscated model to original model: {e}")
            pass

        logger.info(f"⬇️ Received {len(valid_miner_responses)} valid responses")
        if valid_miner_responses is None or len(valid_miner_responses) == 0:
            logger.info("No valid miner responses to process... skipping")
            return

        # include the ground_truth to keep in data manager
        synapse.ground_truth = data.ground_truth
        synapse.dendrite.hotkey = self.wallet.hotkey.ss58_address
        response_data = DendriteQueryResponse(
            request=synapse,
            miner_responses=valid_miner_responses,
        )

        logger.debug("Attempting to saving dendrite response")
        vali_request_model = await ORM.save_task(
            validator_request=synapse,
            miner_responses=valid_miner_responses,
            ground_truth=data.ground_truth,
        )

        if vali_request_model is None:
            logger.error("Failed to save dendrite response")
            return

        # saving response
        logger.success(
            f"Saved dendrite response for request id: {response_data.request.request_id}"
        )
        logger.info(
            f"Sending request to miners & processing took {get_epoch_time() - start}"
        )
        return

    @staticmethod
    async def _send_shuffled_requests(
            dendrite: bt.dendrite, axons: List[bt.AxonInfo], synapse: FeedbackRequest
    ) -> list[FeedbackRequest]:
        """Send the same request to all miners without shuffling the order.
         WARNING: This should only be used for testing/debugging as it could allow miners to game the system.

         Args:
             dendrite (bt.dendrite): Communication channel to send requests
             axons (List[bt.AxonInfo]): List of miner endpoints
             synapse (FeedbackRequest): The feedback request to send

         Returns:
             list[FeedbackRequest]: List of miner responses
         """
        all_responses = []
        batch_size = 10

        for i in range(0, len(axons), batch_size):
            batch_axons = axons[i: i + batch_size]
            tasks = []

            for axon in batch_axons:
                tasks.append(
                    dendrite.forward(
                        axons=[axon],
                        synapse=synapse,
                        deserialize=False,
                        timeout=12,
                    )
                )

            batch_responses = await asyncio.gather(*tasks)
            flat_batch_responses = [
                response for sublist in batch_responses for response in sublist
            ]
            all_responses.extend(flat_batch_responses)

            logger.info(
                f"Processed batch {i // batch_size + 1} of {(len(axons) - 1) // batch_size + 1}"
            )

        return all_responses
