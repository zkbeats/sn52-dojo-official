import asyncio
import copy
import gc
import random
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from traceback import print_exception
from typing import List

import aiohttp
import bittensor as bt
import numpy as np
import torch
import wandb
from bittensor.btlogging import logging as logger
from fastapi.encoders import jsonable_encoder
from tenacity import RetryError
from torch.nn import functional as F

import dojo
from commons.dataset.synthetic import SyntheticAPI
from commons.exceptions import (
    EmptyScores,
    InvalidMinerResponse,
    NoNewUnexpiredTasksYet,
    SetWeightsFailed,
)
from commons.obfuscation.obfuscation_utils import obfuscate_html_and_js
from commons.orm import ORM
from commons.scoring import Scoring
from commons.utils import (
    datetime_as_utc,
    get_epoch_time,
    get_new_uuid,
    init_wandb,
    set_expire_time,
)
from database.client import connect_db
from dojo.base.neuron import BaseNeuron
from dojo.protocol import (
    CompletionResponses,
    CriteriaTypeEnum,
    DendriteQueryResponse,
    FeedbackRequest,
    Heartbeat,
    MultiScoreCriteria,
    ScoringResult,
    TaskResult,
    TaskResultRequest,
    TaskType,
)
from dojo.utils.config import get_config
from dojo.utils.uids import MinerUidSelector, extract_miner_uids, is_miner


class Validator(BaseNeuron):
    _should_exit: bool = False
    _alock = asyncio.Lock()
    _tlock = threading.Lock()
    _threshold = 0.1
    _active_miner_uids: set[int] = set()

    def __init__(self):
        super().__init__()
        self.loop = asyncio.get_event_loop()

        # Dendrite lets us send messages to other nodes (axons) in the network.
        self.dendrite = bt.dendrite(wallet=self.wallet)
        logger.info(f"Dendrite: {self.dendrite}")
        # Set up initial scoring weights for validation
        self.scores: torch.Tensor = torch.zeros(
            self.metagraph.n.item(), dtype=torch.float32
        )
        self.load_state()

        # manually always register and always sync metagraph when application starts
        self.check_registered()
        self.resync_metagraph()
        self.executor = ThreadPoolExecutor(max_workers=2)

        init_wandb(config=self.config, my_uid=self.uid, wallet=self.wallet)

    async def send_scores(self, synapse: ScoringResult, hotkeys: List[str]):
        """Send consensus score back to miners who participated in the request."""
        axons = [axon for axon in self.metagraph.axons if axon.hotkey in hotkeys]
        if not axons:
            logger.warning("No axons to send consensus to... skipping")
        else:
            logger.debug(
                f"Sending back consensus to miners for request id: {synapse.request_id}"
            )

        await self.dendrite.forward(
            axons=axons, synapse=synapse, deserialize=False, timeout=12
        )

    async def update_score_and_send_feedback(self):
        while True:
            await asyncio.sleep(dojo.VALIDATOR_UPDATE_SCORE)
            logger.info("📝 performing scoring ...")
            try:
                validator_hotkeys = [
                    hotkey
                    for uid, hotkey in enumerate(self.metagraph.hotkeys)
                    if not is_miner(self.metagraph, uid)
                ]

                if get_config().ignore_min_stake:
                    validator_hotkeys.append(self.wallet.hotkey.ss58_address)

                batch_id = 0
                # number of tasks to process in a batch
                batch_size = 10
                processed_request_ids = []

                # figure out an expire_at cutoff time to determine those requests ready for scoring
                try:
                    expire_at = await ORM.get_last_expire_at_cutoff(validator_hotkeys)
                except ValueError:
                    logger.warning(
                        f"No tasks for scoring yet, please wait for tasks to to pass deadline of {dojo.TASK_DEADLINE} seconds"
                    )
                    continue

                async for (
                    task_batch,
                    has_more_batches,
                ) in ORM.get_expired_tasks(
                    validator_hotkeys, batch_size=batch_size, expire_at=expire_at
                ):
                    if not has_more_batches:
                        logger.success(
                            f"📝 All tasks processed, total batches: {batch_id}, batch size: {batch_size}"
                        )
                        gc.collect()
                        break

                    if not task_batch:
                        break

                    batch_id += 1
                    logger.info(
                        f"📝 Processing batch {batch_id}, batch size: {batch_size}"
                    )
                    for task in task_batch:
                        criteria_to_miner_score, hotkey_to_score = (
                            Scoring.calculate_score(
                                criteria_types=task.request.criteria_types,
                                request=task.request,
                                miner_responses=task.miner_responses,
                            )
                        )
                        logger.debug(f"📝 Got hotkey to score: {hotkey_to_score}")
                        logger.debug(
                            f"📝 Initially had {len(task.miner_responses)} responses from miners, but only {len(hotkey_to_score.keys())} valid responses"
                        )

                        if not hotkey_to_score:
                            logger.info(
                                "📝 Did not manage to generate a dict of hotkey to score"
                            )
                            # append it anyways so we can cut off later
                            processed_request_ids.append(task.request.request_id)
                            continue

                        with self._tlock:
                            self.update_scores(hotkey_to_scores=hotkey_to_score)
                        await self.send_scores(
                            synapse=ScoringResult(
                                request_id=task.request.request_id,
                                hotkey_to_scores=hotkey_to_score,
                            ),
                            hotkeys=list(hotkey_to_score.keys()),
                        )

                        async def log_wandb():
                            # calculate mean across all criteria

                            if (
                                not criteria_to_miner_score.values()
                                or not hotkey_to_score
                            ):
                                logger.warning(
                                    "📝 No criteria to miner scores available. Skipping calculating averages for wandb."
                                )
                                return

                            mean_weighted_consensus_scores = (
                                torch.stack(
                                    [
                                        miner_scores.consensus.score
                                        for miner_scores in criteria_to_miner_score.values()
                                    ]
                                )
                                .mean(dim=0)
                                .tolist()
                            )
                            mean_weighted_gt_scores = (
                                torch.stack(
                                    [
                                        miner_scores.ground_truth
                                        for miner_scores in criteria_to_miner_score.values()
                                    ]
                                )
                                .mean(dim=0)
                                .tolist()
                            )

                            logger.info(
                                f"📝 Mean miner scores across different criteria: consensus shape:{mean_weighted_consensus_scores}, gt shape:{mean_weighted_gt_scores}"
                            )

                            score_data = {}
                            # update the scores based on the rewards
                            score_data["scores_by_hotkey"] = hotkey_to_score
                            score_data["mean"] = {
                                "consensus": mean_weighted_consensus_scores,
                                "ground_truth": mean_weighted_gt_scores,
                            }

                            wandb_data = jsonable_encoder(
                                {
                                    "task": task.request.task_type,
                                    "criteria": task.request.criteria_types,
                                    "prompt": task.request.prompt,
                                    "completions": jsonable_encoder(
                                        task.request.completion_responses
                                    ),
                                    "num_completions": len(
                                        task.request.completion_responses
                                    ),
                                    "scores": score_data,
                                    "num_responses": len(task.miner_responses),
                                }
                            )

                            wandb.log(wandb_data, commit=True)

                        asyncio.create_task(log_wandb())

                        # once we have scored a response, just remove it
                        processed_request_ids.append(task.request.request_id)

                if processed_request_ids:
                    await ORM.mark_tasks_processed_by_request_ids(processed_request_ids)

            except Exception:
                traceback.print_exc()
                pass

    def obfuscate_model_names(
        self, completion_responses: list[CompletionResponses]
    ) -> dict[str, str]:
        """Obfuscate model names for both external requests and synthetic requests to prevent miners from knowing the true model names."""
        obfuscated_model_to_model: dict[str, str] = {}
        for completion in completion_responses:
            if completion.completion_id is None:
                raise ValueError("completion_id is None")
            completion.model = completion.completion_id
            obfuscated_model_to_model[completion.completion_id] = (
                completion.completion_id
            )
        return obfuscated_model_to_model

    async def send_heartbeats(self):
        """Perform a health check periodically to ensure miners are reachable"""
        while True:
            await asyncio.sleep(dojo.VALIDATOR_HEARTBEAT)
            self.resync_metagraph()
            try:
                all_miner_uids = extract_miner_uids(metagraph=self.metagraph)
                logger.debug(f"Sending heartbeats to {len(all_miner_uids)} miners")
                axons: list[bt.AxonInfo] = [
                    self.metagraph.axons[uid]
                    for uid in all_miner_uids
                    if self.metagraph.axons[uid].hotkey.casefold()
                    != self.wallet.hotkey.ss58_address.casefold()
                ]

                responses: List[Heartbeat] = await self.dendrite.forward(  # type: ignore
                    axons=axons, synapse=Heartbeat(), deserialize=False, timeout=12
                )
                active_hotkeys = [r.axon.hotkey for r in responses if r.ack and r.axon]
                active_uids = [
                    uid
                    for uid, axon in enumerate(self.metagraph.axons)
                    if axon.hotkey in active_hotkeys
                ]
                async with self._alock:
                    self._active_miner_uids = set(active_uids)
                logger.debug(
                    f"⬇️ Heartbeats acknowledged by active miners: {sorted(active_uids)}"
                )
            except Exception as e:
                logger.error(
                    f"Error in sending heartbeats: {e}, traceback: {traceback.format_exc()}"
                )
                pass

    @staticmethod
    async def _send_shuffled_requests(
        dendrite: bt.dendrite, axons: List[bt.AxonInfo], synapse: FeedbackRequest
    ) -> list[FeedbackRequest]:
        """Based on the initial synapse, send shuffled ordering of responses so that miners cannot guess ordering of ground truth"""
        tasks = []
        for axon in axons:
            # shuffle synapse Responses
            shuffled_completions = random.sample(
                synapse.completion_responses,
                k=len(synapse.completion_responses),
            )

            # Apply obfuscation to each completion's files
            # TODO re-nable obfuscation
            # await Validator._obfuscate_completion_files(shuffled_completions)

            criteria_types = []
            # ensure criteria options same order as completion_responses
            for criteria in synapse.criteria_types:
                if not isinstance(criteria, MultiScoreCriteria):
                    logger.trace(f"Skipping non multi score criteria: {criteria}")
                    continue
                options = [completion.model for completion in shuffled_completions]
                criteria = MultiScoreCriteria(
                    options=options,
                    min=criteria.min,
                    max=criteria.max,
                )
                criteria_types.append(criteria)

            shuffled_synapse = FeedbackRequest(
                epoch_timestamp=synapse.epoch_timestamp,
                request_id=synapse.request_id,
                prompt=synapse.prompt,
                completion_responses=shuffled_completions,
                task_type=synapse.task_type,
                criteria_types=criteria_types,
                expire_at=synapse.expire_at,
            )

            tasks.append(
                dendrite.forward(
                    axons=[axon],
                    synapse=shuffled_synapse,
                    deserialize=False,
                    timeout=12,
                )
            )

        # Gather results and flatten the list
        nested_responses = await asyncio.gather(*tasks)
        flat_responses = [
            response for sublist in nested_responses for response in sublist
        ]

        return flat_responses

    @staticmethod
    async def _obfuscate_completion_files(
        completion_responses: List[CompletionResponses],
    ):
        """Obfuscate HTML files in each completion response."""
        for completion in completion_responses:
            if hasattr(completion.completion, "files"):
                for file in completion.completion.files:
                    if file.filename.lower().endswith(".html"):
                        try:
                            original_size = len(file.content)
                            logger.debug(
                                f"Original size of {file.filename}: {original_size} bytes"
                            )
                            file.content = await obfuscate_html_and_js(file.content)
                            obfuscated_size = len(file.content)
                            logger.debug(
                                f"Obfuscated size of {file.filename}: {obfuscated_size} bytes"
                            )
                        except Exception as e:
                            logger.error(f"Error obfuscating {file.filename}: {e}")

    async def get_miner_uids(self, is_external_request: bool, request_id: str):
        async with self._alock:
            if is_external_request:
                sel_miner_uids = [
                    uid
                    for uid in self._active_miner_uids
                    if self.scores[uid] > self._threshold
                ]
                logger.debug(
                    f"🌍 External user request, number of miners with scores above threshold: {len(sel_miner_uids)}"
                )
            else:
                sel_miner_uids = MinerUidSelector(
                    nodes=list(self._active_miner_uids),
                ).get_target_uids(key=request_id, k=get_config().neuron.sample_size)
        return sel_miner_uids

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
        sel_miner_uids = await self.get_miner_uids(external_user, request_id)
        # TODO @dev REMOVE AFTER TESTING
        # TODO @dev REMOVE AFTER TESTING
        # TODO @dev REMOVE AFTER TESTING
        # TODO @dev REMOVE AFTER TESTING
        # TODO @dev REMOVE AFTER TESTING
        # TODO @dev REMOVE AFTER TESTING
        # TODO @dev REMOVE AFTER TESTING
        # TODO @dev REMOVE AFTER TESTING
        # TODO @dev REMOVE AFTER TESTING
        # TODO @dev REMOVE AFTER TESTING
        # TODO @dev REMOVE AFTER TESTING
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
                logger.debug(
                    f"Received response from miner: {miner_hotkey, miner_response.dojo_task_id}"
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

                logger.debug(
                    f"Successfully mapped obfuscated model names for {miner_hotkey}"
                )

                # update the miner response with the real model ids
                valid_miner_responses.append(miner_response)
        except Exception as e:
            logger.error(f"Failed to map obfuscated model to original model: {e}")
            pass

        logger.info(f"⬇️ Got {len(valid_miner_responses)} valid responses")

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

    async def run(self):
        logger.info(f"Validator starting at block: {str(self.block)}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                try:
                    await self.send_request()

                    # # Check if we should exit.
                    if self._should_exit:
                        logger.debug("Validator should stop...")
                        break

                    # Sync metagraph and potentially set weights.
                    await self.loop.run_in_executor(self.executor, self.sync)

                    self.step += 1
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f"Error during validator run: {e}")
                    pass
                await asyncio.sleep(dojo.VALIDATOR_RUN)

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            logger.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            logger.error("Error during validation", str(err))
            logger.debug(print_exception(type(err), err, err.__traceback__))

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners.
        The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """

        # ensure self.scores not being written to by other coroutines
        # Check if self.scores contains any NaN values and log a warning if it does.
        if torch.isnan(self.scores).any():
            logger.warning(
                "Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        logger.info("Attempting to set weights")

        safe_uids = self.metagraph.uids
        if isinstance(self.metagraph.uids, np.ndarray):
            safe_uids = torch.from_numpy(safe_uids).to("cpu")
        elif isinstance(self.metagraph.uids, torch.Tensor):
            pass

        # ensure sum = 1
        normalized_weights = F.normalize(self.scores.cpu(), p=1, dim=0)

        safe_normalized_weights = normalized_weights
        if isinstance(normalized_weights, np.ndarray):
            safe_normalized_weights = torch.from_numpy(normalized_weights).to("cpu")
        elif isinstance(normalized_weights, torch.Tensor):
            pass

        min_allowed_weights = self.subtensor.min_allowed_weights(self.config.netuid)
        max_weight_limit = self.subtensor.max_weight_limit(self.config.netuid)
        logger.debug(f"min_allowed_weights: {min_allowed_weights}")
        logger.debug(f"max_weight_limit: {max_weight_limit}")

        logger.debug("Attempting to set weights")
        logger.debug(f"weights: {safe_normalized_weights}")
        logger.debug(f"uids: {safe_uids}")
        self.set_weights_in_thread(safe_uids, safe_normalized_weights)
        return

    def set_weights_in_thread(self, uids: torch.Tensor, weights: torch.Tensor):
        """Wrapper function to set weights in a separate thread

        Args:
            uids (torch.Tensor): uids to set weights for
            weights (torch.Tensor): weights to set

        Returns:
            tuple[bool, str]: Returns the result of _set_weights function
        """
        logger.trace("Attempting to set weights in another thread")

        def _set_weights(lock: threading.Lock) -> tuple[bool, str]:
            """LOCAL FUNCTION to set weights, we pass in a lock because of how
            we are calling this function from the main thread, sending it
            to a separate thread to avoid blocking the main thread, so the lock
            MUST be acquired by the separate thread.


            Args:
                lock (threading.Lock): Lock parameter passed to separate thread

            Returns:
                tuple[bool, str]: Returns a tuple of a boolean and a string
                - boolean: True if weights were set successfully, False otherwise
                - string: Message indicating the result of set weights
            """
            with lock:
                max_attempts = 5
                attempt = 0
                while attempt < max_attempts:
                    try:
                        logger.trace(f"Set weights attempt {attempt+1}/{max_attempts}")
                        result, message = self.subtensor.set_weights(
                            wallet=self.wallet,
                            netuid=self.config.netuid,  # type: ignore
                            uids=uids.tolist(),
                            weights=weights.tolist(),
                            wait_for_finalization=False,
                            wait_for_inclusion=False,
                            version_key=self.spec_version,
                            max_retries=1,
                        )
                        if result:
                            logger.success(f"Set weights successfully: {message}")
                            return result, message

                        logger.warning(
                            f"Failed to set weights with attempt {attempt+1}/{max_attempts} due to: {message}"
                        )
                        raise SetWeightsFailed(
                            f"Failed to set weights with message:{message}"
                        )

                    except Exception as e:
                        attempt += 1
                        logger.warning(f"Attempt {attempt} failed: {e}")
                        if attempt == max_attempts:
                            logger.error("Max attempts reached. Could not set weights.")
                            return False, "Max attempts reached"

                        self._wait_set_weights()

            return False, "Max attempts reached"

        logger.trace("Submitting callable func to executor")
        future = self.executor.submit(_set_weights, self._tlock)
        result = future.result()
        return result

    def _wait_set_weights(self):
        """Waits for 1 block by calling the block number. Otherwise waits until 24s"""
        logger.trace("Waiting for 1 block before setting weights")
        current_block = self.block
        start_time = time.time()
        while self.block == current_block:
            # long max wait before retrying, up to 2 blocks
            if time.time() - start_time > 2 * 12:
                logger.warning("Waited for 1 block before setting weights, retrying...")
                break
            time.sleep(3)

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        logger.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(previous_metagraph.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(previous_metagraph.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = torch.zeros(self.metagraph.n)
            min_len = min(len(previous_metagraph.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            with self._tlock:
                self.scores = new_moving_average

    def update_scores(self, hotkey_to_scores: dict[str, float]):
        """Performs exponential moving average on the scores based on the rewards received from the miners,
        after setting the self.scores variable here, `set_weights` will be called to set the weights on chain.
        """
        if not hotkey_to_scores:
            logger.warning("hotkey_to_scores is empty, skipping score update")
            return

        nan_value_indices = np.isnan(list(hotkey_to_scores.values()))
        if nan_value_indices.any():
            logger.warning(f"NaN values detected in rewards: {hotkey_to_scores}")
            return

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # scores dimensions might have been updated after resyncing... len(uids) != len(self.scores)
        rewards = torch.zeros((len(self.metagraph.axons),))
        neuron_hotkeys: List[str] = [neuron.hotkey for neuron in self.metagraph.neurons]
        for index, (key, value) in enumerate(hotkey_to_scores.items()):
            # handle nan values
            if nan_value_indices[index]:
                rewards[key] = 0.0  # type: ignore
            # search metagraph for hotkey and grab uid
            try:
                uid = neuron_hotkeys.index(key)
            except ValueError:
                logger.warning(
                    "Old hotkey found from previous metagraph, skip setting weights"
                )
                continue

            logger.debug(f"Score for hotkey {key} is {value}")
            rewards[uid] = value

        logger.debug(f"Rewards: {rewards}")
        # Update scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        alpha: float = self.config.neuron.moving_average_alpha
        # don't acquire lock here because we're already acquiring it in the CALLER
        self.scores = alpha * rewards + (1 - alpha) * self.scores
        logger.debug(f"Updated scores: {self.scores}")

    async def _save_state(
        self,
    ):
        """Saves the state of the validator to the database."""
        if self.step == 0:
            return

        try:
            if np.count_nonzero(self.scores) == 0:
                logger.warning("Scores are all zeros, but saving anyway!")
                # raise EmptyScores("Skipping save as scores are all empty")

            await ORM.create_or_update_validator_score(self.scores)
            logger.success(f"📦 Saved validator state with scores: {self.scores}")
        except EmptyScores as e:
            logger.debug(f"No need to to save validator state: {e}")
        except Exception as e:
            logger.error(f"Failed to save validator state: {e}")

    def save_state(self):
        try:
            self.loop.run_until_complete(self._save_state())
        except Exception as e:
            logger.error(f"Failed to save validator state: {e}")

    async def _load_state(self):
        try:
            await connect_db()
            scores = await ORM.get_validator_score()

            if scores is None:
                num_processed_tasks = await ORM.get_num_processed_tasks()
                if num_processed_tasks > 0:
                    logger.error(
                        "Score record not found, but you have processed tasks."
                    )
                else:
                    logger.warning(
                        "Score record not found, and no tasks processed, this is okay if you're running for the first time."
                    )
                return None

            logger.success(f"Loaded validator state: {scores=}")
            with self._tlock:
                self.scores = scores

        except Exception as e:
            logger.error(
                f"Unexpected error occurred while loading validator state: {e}"
            )
            return None

    def load_state(self):
        """Loads the state of the validator from a file."""
        try:
            self.loop.run_until_complete(self._load_state())
        except Exception as e:
            logger.error(f"Failed to load validator state: {e}")
            pass

    async def log_validator_status(self):
        while not self._should_exit:
            logger.info(
                f"Validator running... block:{str(self.block)} time: {time.time()}"
            )
            await asyncio.sleep(dojo.VALIDATOR_STATUS)

    async def _get_task_results_from_miner(
        self, miner_hotkey: str, task_id: str
    ) -> list[TaskResult]:
        """Fetch task results from the miner's Axon using Dendrite."""
        try:
            if not self.dendrite:
                raise ValueError("Dendrite not initialized")

            # Prepare the synapse (data request) that will be sent via Dendrite
            task_synapse = TaskResultRequest(task_id=task_id)

            # Use Dendrite to communicate with the Axon
            miner_axon = self.metagraph.axons[
                self.metagraph.hotkeys.index(miner_hotkey)
            ]
            if not miner_axon:
                raise ValueError(f"Miner Axon not found for hotkey: {miner_hotkey}")

            # Send the request via Dendrite and get the response
            response: list[TaskResultRequest] = await self.dendrite.forward(  # type: ignore
                axons=[miner_axon], synapse=task_synapse, deserialize=False
            )

            if response and response[0]:
                logger.debug(
                    f"Received task result from miner {miner_hotkey} for task {task_id}, {response}"
                )
                return response[0].task_results
            else:
                logger.debug(
                    f"No task results found from miner {miner_hotkey} for task {task_id}"
                )
                return []

        except Exception as e:
            logger.error(f"Error fetching task result from miner {miner_hotkey}: {e}")
            return []

    async def monitor_task_completions(self):
        while not self._should_exit:
            try:
                validator_hotkeys = [
                    hotkey
                    for uid, hotkey in enumerate(self.metagraph.hotkeys)
                    if not is_miner(self.metagraph, uid)
                ]

                if get_config().ignore_min_stake:
                    validator_hotkeys.append(self.wallet.hotkey.ss58_address)

                batch_id = 0
                batch_size = 10
                # use current time as cutoff so we get only unexpired tasks
                now = datetime_as_utc(datetime.now(timezone.utc))
                async for task_batch, has_more_batches in ORM.get_expired_tasks(
                    validator_hotkeys=validator_hotkeys,
                    batch_size=batch_size,
                    expire_at=now,
                ):
                    if not has_more_batches:
                        logger.success(
                            "No more unexpired tasks found for processing, exiting task monitoring."
                        )
                        gc.collect()
                        break

                    if not task_batch:
                        continue

                    batch_id += 1
                    logger.info(f"Monitoring task completions, batch id: {batch_id}")

                    for task in task_batch:
                        request_id = task.request.request_id
                        miner_responses = task.miner_responses

                        obfuscated_to_real_model_id = await ORM.get_real_model_ids(
                            request_id
                        )

                        for miner_response in miner_responses:
                            if (
                                not miner_response.axon
                                or not miner_response.axon.hotkey
                                or not miner_response.dojo_task_id
                            ):
                                raise InvalidMinerResponse(
                                    f"""Missing hotkey, task_id, or axon:
                                    axon: {miner_response.axon}
                                    hotkey: {miner_response.axon.hotkey}
                                    task_id: {miner_response.dojo_task_id}"""
                                )

                            miner_hotkey = miner_response.axon.hotkey
                            task_id = miner_response.dojo_task_id
                            task_results = await asyncio.create_task(
                                self._get_task_results_from_miner(miner_hotkey, task_id)
                            )

                            if not task_results and not len(task_results) > 0:
                                logger.debug(
                                    f"Task ID: {task_id} by miner: {miner_hotkey} has not been completed yet or no task results."
                                )
                                continue

                            # Process task result
                            model_id_to_avg_rank, model_id_to_avg_score = (
                                self._calculate_averages(
                                    task_results, obfuscated_to_real_model_id
                                )
                            )

                            # Update the response with the new ranks and scores
                            for completion in miner_response.completion_responses:
                                model_id = completion.model
                                if model_id in model_id_to_avg_rank:
                                    completion.rank_id = int(
                                        model_id_to_avg_rank[model_id]
                                    )
                                if model_id in model_id_to_avg_score:
                                    completion.score = model_id_to_avg_score[model_id]

                            # Update miner responses in the database
                            success = await ORM.update_miner_completions_by_request_id(
                                request_id, task.miner_responses
                            )

                            logger.info(
                                f"Updating task {request_id} with miner's completion data, success ? {success}"
                            )
                        await asyncio.sleep(0.2)
            except NoNewUnexpiredTasksYet as e:
                logger.info(f"No new unexpired tasks yet: {e}")
            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error during Dojo task monitoring {str(e)}")
                pass
            await asyncio.sleep(dojo.DOJO_TASK_MONITORING)

    @staticmethod
    def _calculate_averages(
        task_results: list[TaskResult], obfuscated_to_real_model_id
    ):
        model_id_to_avg_rank = defaultdict(float)
        model_id_to_avg_score = defaultdict(float)
        num_ranks_by_workers, num_scores_by_workers = 0, 0

        for result in task_results:
            for result_data in result.result_data:
                type = result_data.type
                value = result_data.value
                if type == CriteriaTypeEnum.RANKING_CRITERIA:
                    for model_id, rank in value.items():
                        real_model_id = obfuscated_to_real_model_id.get(
                            model_id, model_id
                        )
                        model_id_to_avg_rank[real_model_id] += rank
                    num_ranks_by_workers += 1
                elif type == CriteriaTypeEnum.MULTI_SCORE:
                    for model_id, score in value.items():
                        real_model_id = obfuscated_to_real_model_id.get(
                            model_id, model_id
                        )
                        model_id_to_avg_score[real_model_id] += score
                    num_scores_by_workers += 1

        # Average the ranks and scores
        for model_id in model_id_to_avg_rank:
            model_id_to_avg_rank[model_id] /= num_ranks_by_workers
        for model_id in model_id_to_avg_score:
            model_id_to_avg_score[model_id] /= num_scores_by_workers

        return model_id_to_avg_rank, model_id_to_avg_score
