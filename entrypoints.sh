#!/bin/bash

set -e

# run bash
if [ "$1" = 'btcli' ]; then
    exec /bin/bash -c "btcli --help && exec /bin/bash"
fi

# run dojo cli
if [ "$1" = 'dojo-cli' ]; then
    dojo
fi

if [ "$1" = 'miner' ]; then
    echo "Environment variables:"
    echo "WALLET_COLDKEY: ${WALLET_COLDKEY}"
    echo "WALLET_HOTKEY: ${WALLET_HOTKEY}"
    echo "AXON_PORT: ${AXON_PORT}"
    echo "SUBTENSOR_NETWORK: ${SUBTENSOR_NETWORK}"
    echo "SUBTENSOR_ENDPOINT: ${SUBTENSOR_ENDPOINT}"
    echo "NETUID: ${NETUID}"

    EXTRA_ARGS=""
    if [ "${SIMULATION}" = "true" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --simulation"
    fi
    if [ "${FAST_MODE}" = "true" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --fast_mode"
    fi
    if [ "${SIMULATION_BAD_MINER}" = "true" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --simulation_bad_miner"
    fi

    python main_miner.py \
    --netuid ${NETUID} \
    --subtensor.network ${SUBTENSOR_NETWORK} \
    --subtensor.chain_endpoint ${SUBTENSOR_ENDPOINT} \
    --logging.debug \
    --wallet.name ${WALLET_COLDKEY} \
    --wallet.hotkey ${WALLET_HOTKEY} \
    --axon.port ${AXON_PORT} \
    --neuron.type miner \
    ${EXTRA_ARGS}
fi

# If the first argument is 'validator', run the validator script
if [ "$1" = 'validator' ]; then
    echo "Environment variables:"
    echo "WALLET_COLDKEY: ${WALLET_COLDKEY}"
    echo "WALLET_HOTKEY: ${WALLET_HOTKEY}"
    echo "AXON_PORT: ${AXON_PORT}"
    echo "SUBTENSOR_NETWORK: ${SUBTENSOR_NETWORK}"
    echo "SUBTENSOR_ENDPOINT: ${SUBTENSOR_ENDPOINT}"
    echo "NETUID: ${NETUID}"
    echo "WANDB_PROJECT_NAME: ${WANDB_PROJECT_NAME}"

    EXTRA_ARGS=""
    if [ "${SIMULATION}" = "true" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --simulation"
    fi
    if [ "${FAST_MODE}" = "true" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --fast_mode"
    fi

    python main_validator.py \
    --netuid ${NETUID} \
    --subtensor.network ${SUBTENSOR_NETWORK} \
    --subtensor.chain_endpoint ${SUBTENSOR_ENDPOINT} \
    --logging.debug \
    --wallet.name ${WALLET_COLDKEY} \
    --wallet.hotkey ${WALLET_HOTKEY} \
    --neuron.type validator \
    --wandb.project_name ${WANDB_PROJECT_NAME} \
    ${EXTRA_ARGS}
fi