import json
from datetime import timedelta

import bittensor as bt
from loguru import logger

import template
from commons.utils import is_valid_expiry, set_expire_time
from database.prisma import Json
from database.prisma.enums import CriteriaTypeEnumModel
from database.prisma.models import CriteriaTypeModel, FeedbackRequestModel
from database.prisma.types import (
    CompletionResponseModelCreateInput,
    CriteriaTypeModelCreateInput,
    FeedbackRequestModelCreateInput,
    MinerResponseModelCreateInput,
)
from template.protocol import (
    CompletionResponses,
    CriteriaType,
    DendriteQueryResponse,
    FeedbackRequest,
    MultiScoreCriteria,
    MultiSelectCriteria,
    RankingCriteria,
    ScoreCriteria,
)


def map_criteria_type_to_model(
    criteria: CriteriaType, request_id: str
) -> CriteriaTypeModelCreateInput:
    try:
        if isinstance(criteria, RankingCriteria):
            return CriteriaTypeModelCreateInput(
                type=CriteriaTypeEnumModel.RANKING_CRITERIA,
                request_id=request_id,
                # options=cast(Json, json.dumps(criteria.options)),
                options=Json(json.dumps(criteria.options)),
            )
        elif isinstance(criteria, ScoreCriteria):
            return CriteriaTypeModelCreateInput(
                type=CriteriaTypeEnumModel.SCORE,
                request_id=request_id,
                min=criteria.min,
                max=criteria.max,
            )
        elif isinstance(criteria, MultiSelectCriteria):
            return CriteriaTypeModelCreateInput(
                type=CriteriaTypeEnumModel.MULTI_SELECT,
                request_id=request_id,
                options=Json(json.dumps(criteria.options)),
            )
        elif isinstance(criteria, MultiScoreCriteria):
            return CriteriaTypeModelCreateInput(
                type=CriteriaTypeEnumModel.MULTI_SCORE,
                request_id=request_id,
                options=Json(json.dumps(criteria.options)),
                min=criteria.min,
                max=criteria.max,
            )
        else:
            raise ValueError("Unknown criteria type")
    except Exception as e:
        raise ValueError(f"Failed to map criteria type to model {e}")


def map_criteria_type_model_to_criteria_type(model: CriteriaTypeModel) -> CriteriaType:
    try:
        if model.type == CriteriaTypeEnumModel.RANKING_CRITERIA:
            return RankingCriteria(
                options=json.loads(model.options) if model.options else []
            )
        elif model.type == CriteriaTypeEnumModel.SCORE:
            return ScoreCriteria(
                min=model.min if model.min is not None else 0.0,
                max=model.max if model.max is not None else 0.0,
            )
        elif model.type == CriteriaTypeEnumModel.MULTI_SELECT:
            return MultiSelectCriteria(
                options=json.loads(model.options) if model.options else []
            )
        elif model.type == CriteriaTypeEnumModel.MULTI_SCORE:
            return MultiScoreCriteria(
                options=json.loads(model.options) if model.options else [],
                min=model.min if model.min is not None else 0.0,
                max=model.max if model.max is not None else 0.0,
            )
        else:
            raise ValueError("Unknown criteria type")
    except Exception as e:
        logger.error(f"Failed to map criteria type model to criteria type: {e}")
        raise ValueError("Failed to map criteria type model to criteria type")


def map_completion_response_to_model(
    response: CompletionResponses, miner_response_id: str
) -> CompletionResponseModelCreateInput:
    try:
        result = CompletionResponseModelCreateInput(
            completion_id=response.completion_id,
            model=response.model,
            # completion=cast(Json, json.dumps(response.completion)),
            completion=Json(json.dumps(response.completion, default=vars)),
            rank_id=response.rank_id,
            score=response.score,
            miner_response_id=miner_response_id,
        )
        return result
    except Exception as e:
        raise ValueError(f"Failed to map completion response to model {e}")


def map_miner_response_to_model(
    response: FeedbackRequest, request_id: str
) -> MinerResponseModelCreateInput:
    try:
        # Ensure expire_at is set and is reasonable, this will prevent exploits where miners can set their own expiry times
        expire_at = response.expire_at
        if expire_at is None or is_valid_expiry(expire_at) is not True:
            expire_at = set_expire_time(template.TASK_DEADLINE)

        if response.dojo_task_id is None:
            raise ValueError("Dojo task id is required")

        result = MinerResponseModelCreateInput(
            request_id=request_id,
            miner_hotkey=response.axon.hotkey
            if response.axon and response.axon.hotkey
            else "",
            dojo_task_id=response.dojo_task_id,
            expire_at=expire_at,
        )

        return result
    except Exception as e:
        raise ValueError(f"Failed to map miner response to model {e}")


def map_feedback_request_to_model(
    request: FeedbackRequest,
) -> FeedbackRequestModelCreateInput:
    try:
        result = FeedbackRequestModelCreateInput(
            request_id=request.request_id,
            task_type=request.task_type,
            prompt=request.prompt,
            ground_truth=Json(json.dumps(request.ground_truth)),
        )

        return result
    except Exception as e:
        raise ValueError(f"Failed to map feedback request to model {e}")


def map_model_to_dendrite_query_response(
    model: FeedbackRequestModel,
) -> DendriteQueryResponse:
    try:
        criteria_types = [
            map_criteria_type_model_to_criteria_type(criteria)
            for criteria in model.criteria_types or []
        ]

        completions: list[CompletionResponses] = []
        if model.miner_responses is not None:
            completions = [
                CompletionResponses(
                    completion_id=completion.completion_id,
                    model=completion.model,
                    completion=completion.completion,
                    rank_id=completion.rank_id,
                    score=completion.score,
                )
                for completion in model.miner_responses[0].completions or []
            ]

        # Add TASK_DEADLINE to created_at
        expire_at_dt = model.created_at + timedelta(seconds=template.TASK_DEADLINE)

        request: FeedbackRequest = FeedbackRequest(
            request_id=model.request_id,
            prompt=model.prompt,
            completion_responses=completions,
            task_type=model.task_type,
            criteria_types=criteria_types,
            ground_truth=json.loads(model.ground_truth),
            expire_at=expire_at_dt.isoformat().replace("+00:00", "Z"),
        )

        miner_responses: list[FeedbackRequest] = [
            FeedbackRequest(
                request_id=miner_response.request_id,
                prompt=model.prompt,
                criteria_types=criteria_types,
                task_type=model.task_type,
                dojo_task_id=miner_response.dojo_task_id,
                expire_at=miner_response.expire_at,
                completion_responses=[
                    CompletionResponses(
                        completion_id=completion.completion_id,
                        model=completion.model,
                        completion=completion.completion,
                        rank_id=completion.rank_id,
                        score=completion.score,
                    )
                    for completion in miner_response.completions or []
                ],
                axon=bt.TerminalInfo(hotkey=miner_response.miner_hotkey),
            )
            for miner_response in (model.miner_responses or [])
        ]

        return DendriteQueryResponse(request=request, miner_responses=miner_responses)
    except Exception as e:
        raise ValueError(f"Failed to map model to dendrite query response: {e}")
