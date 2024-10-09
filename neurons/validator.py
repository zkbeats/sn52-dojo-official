import asyncio
import copy
import time
import traceback
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

import template
from commons.data_manager import DataManager, ValidatorStateKeys
from commons.dataset.synthetic import SyntheticAPI
from commons.dojo_task_tracker import DojoTaskTracker
from commons.scoring import Scoring
from commons.utils import get_epoch_time, get_new_uuid, init_wandb, set_expire_time
from template.base.neuron import BaseNeuron
from template.protocol import (
    CompletionResponses,
    DendriteQueryResponse,
    FeedbackRequest,
    Heartbeat,
    MultiScoreCriteria,
    ScoringResult,
    TaskType,
)
from template.utils.config import get_config
from template.utils.uids import MinerUidSelector, extract_miner_uids


class Validator(BaseNeuron):
    _should_exit: bool = False
    _lock = asyncio.Lock()
    _threshold = 0.1
    _active_miner_uids: set[int] = set()

    def __init__(self):
        super().__init__()

        # Dendrite lets us send messages to other nodes (axons) in the network.
        self.dendrite = bt.dendrite(wallet=self.wallet)
        logger.info(f"Dendrite: {self.dendrite}")
        # Set up initial scoring weights for validation
        self.scores = torch.zeros(self.metagraph.n.item(), dtype=torch.float32)
        self.load_state()

        # manually always register and always sync metagraph when application starts
        self.check_registered()
        self.resync_metagraph()

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
        """While this function is triggered every X time period in AsyncIOScheduler,
        only relevant data that has passed the deadline of 8 hours will be scored and sent feedback.
        """
        while True:
            await asyncio.sleep(template.VALIDATOR_UPDATE_SCORE)
            try:
                logger.debug(
                    f"Scheduled update score and send feedback triggered at time: {time.time()}"
                )
                data: List[DendriteQueryResponse] | None = await DataManager.load()
                if not data:
                    logger.debug(
                        "Skipping scoring as no feedback data found, this means either all have been processed or you are running the validator for the first time."
                    )
                    continue

                current_time = datetime.now(timezone.utc)
                # allow enough time for human feedback
                non_expired_data: List[DendriteQueryResponse] = [
                    d
                    for d in data
                    if d.request.expire_at
                    and datetime.fromisoformat(d.request.expire_at) < current_time
                ]
                if not non_expired_data:
                    logger.warning(
                        "Skipping scoring as no feedback data is due for scoring."
                    )

                logger.info(
                    f"Got {len(non_expired_data)} requests past deadline and ready to score"
                )
                for d in non_expired_data:
                    criteria_to_miner_score, hotkey_to_score = Scoring.calculate_score(
                        criteria_types=d.request.criteria_types,
                        request=d.request,
                        miner_responses=d.miner_responses,
                    )
                    logger.trace(f"Got hotkey to score: {hotkey_to_score}")

                    if not hotkey_to_score:
                        request_id = d.request.request_id
                        try:
                            del DojoTaskTracker._rid_to_mhotkey_to_task_id[request_id]
                        except KeyError:
                            logger.warning(
                                f"Failed to remove request id: {request_id} from dojo task tracker"
                            )
                            pass
                        await DataManager.remove_responses([d])
                        continue

                    logger.trace(
                        f"Initially had {len(d.miner_responses)} responses from miners, but only {len(hotkey_to_score.keys())} valid responses"
                    )

                    self.update_scores(hotkey_to_scores=hotkey_to_score)
                    await self.send_scores(
                        synapse=ScoringResult(
                            request_id=d.request.request_id,
                            hotkey_to_scores=hotkey_to_score,
                        ),
                        hotkeys=list(hotkey_to_score.keys()),
                    )

                    # TODO fix why after 5 mins this halts validator.run()
                    async def log_wandb():
                        # calculate mean across all criteria
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
                                    miner_scores.ground_truth.score
                                    for miner_scores in criteria_to_miner_score.values()
                                ]
                            )
                            .mean(dim=0)
                            .tolist()
                        )

                        logger.info(
                            f"mean miner scores across differerent criteria: consensus shape{mean_weighted_consensus_scores.shape}, gt shape:{mean_weighted_gt_scores.shape}"
                        )

                        score_data = {}
                        # update the scores based on the rewards
                        score_data["scores_by_hotkey"] = {
                            hotkey: score.dict()
                            for hotkey, score in hotkey_to_score.items()
                        }
                        score_data["mean"] = {
                            "consensus": mean_weighted_consensus_scores,
                            "ground_truth": mean_weighted_gt_scores,
                        }

                        wandb_data = jsonable_encoder(
                            {
                                "task": d.request.task_type,
                                "criteria": d.request.criteria_types,
                                "prompt": d.request.prompt,
                                "completions": jsonable_encoder(
                                    d.request.completion_responses
                                ),
                                "num_completions": len(d.request.completion_responses),
                                "scores": score_data,
                                "num_responses": len(d.miner_responses),
                            }
                        )

                        wandb.log(wandb_data, commit=True)

                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, log_wandb)

                    # once we have scored a response, just remove it
                    await DataManager.remove_responses([d])

            except Exception:
                traceback.print_exc()
                pass

    def obfuscate_model_names(
        self, completion_responses: list[CompletionResponses]
    ) -> dict[str, str]:
        """Obfuscate model names for both external requests and synthetic requests to prevent miners from knowing the true model names."""
        obfuscated_model_to_model: dict[str, str] = {}
        for completion in completion_responses:
            new_uuid = get_new_uuid()
            obfuscated_model_to_model[new_uuid] = completion.model
            completion.model = new_uuid
        return obfuscated_model_to_model

    async def send_heartbeats(self):
        """Perform a health check periodically to ensure miners are reachable"""
        while True:
            await asyncio.sleep(template.VALIDATOR_HEARTBEAT)
            self.resync_metagraph()
            try:
                all_miner_uids = extract_miner_uids(metagraph=self.metagraph)
                logger.debug(f"⬆️ Sending heartbeats to {len(all_miner_uids)} miners")
                axons: List[bt.AxonInfo] = [
                    self.metagraph.axons[uid]
                    for uid in all_miner_uids
                    if self.metagraph.axons[uid].hotkey.casefold()
                    != self.wallet.hotkey.ss58_address.casefold()
                ]

                responses: List[Heartbeat] = await self.dendrite.forward(
                    axons=axons, synapse=Heartbeat(), deserialize=False, timeout=12
                )
                active_hotkeys = [r.axon.hotkey for r in responses if r.ack and r.axon]
                active_uids = [
                    uid
                    for uid, axon in enumerate(self.metagraph.axons)
                    if axon.hotkey in active_hotkeys
                ]
                async with self._lock:
                    self._active_miner_uids = set(active_uids)
                logger.debug(
                    f"⬇️ Heartbeats acknowledged by active miners: {sorted(active_uids)}"
                )
            except Exception as e:
                logger.error(
                    f"Error in sending heartbeats: {e}, traceback: {traceback.format_exc()}"
                )
                pass

    async def send_request(
        self,
        synapse: FeedbackRequest | None = None,
        external_user: bool = False,
    ):
        start = get_epoch_time()
        # typically the request may come from an external source however,
        # initially will seed it with some data for miners to get started

        # ensure we consider only active miners

        if len(self._active_miner_uids) == 0:
            logger.warning("🤷 No active miners to send request to... skipping")
            return

        request_id = get_new_uuid()
        async with self._lock:
            if external_user:
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
            expire_at = set_expire_time(template.TASK_DEADLINE)
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
            f"⬆️ Sending feedback request for request id: {synapse.request_id}, miners uids:{sel_miner_uids}"
        )

        miner_responses: List[FeedbackRequest] = await self.dendrite.forward(
            axons=axons, synapse=synapse, deserialize=False, timeout=24
        )

        valid_miner_responses: List[FeedbackRequest] = []
        try:
            for miner_response in miner_responses:
                logger.debug(
                    f"Received response from miner: {miner_response.axon.hotkey, miner_response.dojo_task_id}"
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
                    logger.debug(
                        "Miner must provide the dojo task id for scoring method dojo"
                    )
                    continue

                logger.debug(
                    f"Successfully mapped obfuscated model names for {miner_response.axon.hotkey}"
                )

                # update the miner response with the real model ids
                valid_miner_responses.append(miner_response)
        except Exception as e:
            logger.error(f"Failed to map obfuscated model to original model: {e}")
            pass

        logger.info(f"⬇️ Got {len(valid_miner_responses)} valid responses")

        if valid_miner_responses is None or len(valid_miner_responses) == 0:
            logger.warning("No valid miner responses to process... skipping")
            return

        # include the ground_truth to keep in data manager
        synapse.ground_truth = data.ground_truth
        response_data = DendriteQueryResponse(
            request=synapse,
            miner_responses=valid_miner_responses,
        )

        logger.debug("Attempting to saving dendrite response")
        fb_request_model = await DataManager.save_dendrite_response(
            response=response_data
        )

        if fb_request_model is None:
            logger.error("Failed to save dendrite response")
            return

        logger.debug("Attempting to update task map")
        await DojoTaskTracker.update_task_map(
            synapse.request_id,
            fb_request_model,
            obfuscated_model_to_model,
        )

        # saving response
        logger.success(
            f"Saved dendrite response for request id: {response_data.request.request_id}"
        )
        logger.info(
            f"Sending request to miners & processing took {get_epoch_time() - start}"
        )
        return

    async def run(self):
        logger.info(
            f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        logger.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                try:
                    await self.send_request()

                    # # Check if we should exit.
                    if self._should_exit:
                        logger.info("Validator should stop...")
                        break

                    # Sync metagraph and potentially set weights.
                    self.sync()

                    self.step += 1
                except Exception as e:
                    logger.error(f"Error during validator run: {e}")
                    pass
                await asyncio.sleep(template.VALIDATOR_RUN)

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

        # Check if self.scores contains any NaN values and log a warning if it does.
        if torch.isnan(self.scores).any():
            logger.warning(
                "Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        normalized_weights = F.normalize(self.scores.cpu(), p=1, dim=0)

        logger.debug(f"Raw scores: {self.scores}")
        logger.debug(f"normalized weights: {normalized_weights}")
        logger.debug(f"normalized weights uids: {self.metagraph.uids}")

        if torch.count_nonzero(normalized_weights).item() == 0:
            logger.warning("All weights are zero, skipping...")
            return

        logger.info("Attempting to set weights")

        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
            uids=self.metagraph.uids.to("cpu"),
            weights=normalized_weights.to("cpu"),
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        logger.debug(f"processed weights {processed_weights}")
        logger.debug(f"processed weights uids {processed_weight_uids}")

        # Set the weights on chain via our subtensor connection.
        result = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=processed_weight_uids.tolist(),
            weights=processed_weights.tolist(),
            wait_for_finalization=False,
            wait_for_inclusion=True,
            version_key=self.spec_version,
        )
        if result is True:
            logger.success("Validator set weights on chain successfully!")
        else:
            logger.error("set_weights failed")
        return result

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
            new_moving_average = np.zeros(self.metagraph.n)
            min_len = min(len(previous_metagraph.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average

    def update_scores(self, hotkey_to_scores):
        """Performs exponential moving average on the scores based on the rewards received from the miners,
        after setting the self.scores variable here, `set_weights` will be called to set the weights on chain.
        """

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
                rewards[key] = 0.0
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
        self.scores: torch.Tensor = alpha * rewards + (1 - alpha) * self.scores
        logger.debug(f"Updated scores: {self.scores}")

    def save_state(self):
        """Saves the state of the validator to a file."""
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(
                DataManager.validator_save(
                    self.scores,
                    DojoTaskTracker._rid_to_mhotkey_to_task_id,
                    DojoTaskTracker._rid_to_model_map,
                    DojoTaskTracker._task_to_expiry,
                )
            )
        except Exception as e:
            logger.error(f"Failed to save validator state: {e}")
            pass

    def load_state(self):
        """Loads the state of the validator from a file."""
        loop = asyncio.get_event_loop()
        state_data = loop.run_until_complete(DataManager.validator_load())
        if state_data is None:
            logger.error("Failed to load validator state data")
            return

        self.scores = state_data[ValidatorStateKeys.SCORES]
        DojoTaskTracker._rid_to_mhotkey_to_task_id = state_data[
            ValidatorStateKeys.DOJO_TASKS_TO_TRACK
        ]
        DojoTaskTracker._rid_to_model_map = state_data[ValidatorStateKeys.MODEL_MAP]
        DojoTaskTracker._task_to_expiry = state_data[ValidatorStateKeys.TASK_TO_EXPIRY]

        logger.info(f"Scores state: {self.scores}")

    @classmethod
    async def log_validator_status(cls):
        while not cls._should_exit:
            logger.info(f"Validator running... {time.time()}")
            await asyncio.sleep(template.VALIDATOR_STATUS)
