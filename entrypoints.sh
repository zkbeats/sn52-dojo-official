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

if [ "$1" = 'miner-testnet' ]; then
    echo "Environment variables:"
    echo "WALLET_COLDKEY: ${WALLET_COLDKEY}"
    echo "WALLET_HOTKEY: ${WALLET_HOTKEY}"
    echo "AXON_PORT: ${AXON_PORT}"
    echo "SUBTENSOR_NETWORK: ${SUBTENSOR_NETWORK}"
    echo "SUBTENSOR_ENDPOINT: ${SUBTENSOR_ENDPOINT}"

    python main_miner.py \
    --netuid 98 \
    --subtensor.network ${SUBTENSOR_NETWORK} \
    --subtensor.chain_endpoint ${SUBTENSOR_ENDPOINT} \
    --logging.debug \
    --wallet.name ${WALLET_COLDKEY} \
    --wallet.hotkey ${WALLET_HOTKEY} \
    --axon.port ${AXON_PORT} \
    --neuron.type miner \
    --scoring_method dojo
fi

if [ "$1" = 'miner-mainnet' ]; then
    echo "Environment variables:"
    echo "WALLET_COLDKEY: ${WALLET_COLDKEY}"
    echo "WALLET_HOTKEY: ${WALLET_HOTKEY}"
    echo "AXON_PORT: ${AXON_PORT}"
    echo "SUBTENSOR_NETWORK: ${SUBTENSOR_NETWORK}"
    echo "SUBTENSOR_ENDPOINT: ${SUBTENSOR_ENDPOINT}"

    # TODO change netuid before going live
    python main_miner.py \
    --netuid 51 \
    --subtensor.network ${SUBTENSOR_NETWORK} \
    --subtensor.chain_endpoint ${SUBTENSOR_ENDPOINT} \
    --logging.debug \
    --wallet.name ${WALLET_COLDKEY} \
    --wallet.hotkey ${WALLET_HOTKEY} \
    --axon.port ${AXON_PORT} \
    --neuron.type miner
fi

# If the first argument is 'validator', run the validator script
if [ "$1" = 'validator-testnet' ]; then
    echo "Environment variables:"
    echo "WALLET_COLDKEY: ${WALLET_COLDKEY}"
    echo "WALLET_HOTKEY: ${WALLET_HOTKEY}"
    echo "AXON_PORT: ${AXON_PORT}"
    echo "SUBTENSOR_NETWORK: ${SUBTENSOR_NETWORK}"
    echo "SUBTENSOR_ENDPOINT: ${SUBTENSOR_ENDPOINT}"

    python main_validator.py \
    --netuid 98 \
    --subtensor.network ${SUBTENSOR_NETWORK} \
    --subtensor.chain_endpoint ${SUBTENSOR_ENDPOINT} \
    --logging.debug \
    --wallet.name ${WALLET_COLDKEY} \
    --wallet.hotkey ${WALLET_HOTKEY} \
    --neuron.type validator \
    --wandb.project_name dojo-testnet
fi

if [ "$1" = 'validator-mainnet' ]; then
    echo "Environment variables:"
    echo "WALLET_COLDKEY: ${WALLET_COLDKEY}"
    echo "WALLET_HOTKEY: ${WALLET_HOTKEY}"
    echo "AXON_PORT: ${AXON_PORT}"
    echo "SUBTENSOR_NETWORK: ${SUBTENSOR_NETWORK}"
    echo "SUBTENSOR_ENDPOINT: ${SUBTENSOR_ENDPOINT}"

    # TODO change netuid before going live
    python main_validator.py \
    --netuid 51 \
    --subtensor.network ${SUBTENSOR_NETWORK} \
    --subtensor.chain_endpoint ${SUBTENSOR_ENDPOINT} \
    --logging.debug \
    --wallet.name ${WALLET_COLDKEY} \
    --wallet.hotkey ${WALLET_HOTKEY} \
    --neuron.type validator \
    --wandb.project_name dojo-mainnet
fi
