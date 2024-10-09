PRECOMMIT_VERSION="3.7.1"
UNAME := $(shell uname)
.PHONY: hooks btcli install miner-decentralised miner-centralised validator miner-decentralised-logs miner-centralised-logs validator-logs

hooks:
	@echo "Grabbing pre-commit version ${PRECOMMIT_VERSION} and installing pre-commit hooks"
	if [ ! -f pre-commit.pyz ]; then \
		wget -O pre-commit.pyz https://github.com/pre-commit/pre-commit/releases/download/v${PRECOMMIT_VERSION}/pre-commit-${PRECOMMIT_VERSION}.pyz; \
	fi
	python3 pre-commit.pyz clean
	python3 pre-commit.pyz uninstall --hook-type pre-commit --hook-type pre-push --hook-type commit-msg
	python3 pre-commit.pyz gc
	python3 pre-commit.pyz install --hook-type pre-commit --hook-type pre-push --hook-type commit-msg

# ---------------------------------------------------------------------------- #
#                                 INSTALL DEPS                                 #
# ---------------------------------------------------------------------------- #

install:
	@if [ "$(UNAME)" = "Darwin" ]; then \
		pip install -e .; \
	elif [ "$(UNAME)" = "Linux" ]; then \
		pip install -e . --find-links https://download.pytorch.org/whl/torch_stable.html; \
	fi

btcli:
	docker compose -f docker-compose.shared.yaml run --rm btcli

# ---------------------------------------------------------------------------- #
#                                 CORE SERVICES                                #
# ---------------------------------------------------------------------------- #

miner-decentralised:
	@if [ "$(network)" = "mainnet" ]; then \
		docker compose --env-file .env.miner -f docker-compose.miner.yaml up -d --build miner-mainnet-decentralised; \
	elif [ "$(network)" = "testnet" ]; then \
		docker compose --env-file .env.miner -f docker-compose.miner.yaml up -d --build miner-testnet-decentralised; \
	else \
		echo "Please specify a valid network: mainnet or testnet"; \
	fi


miner-centralised:
	@if [ "$(network)" = "mainnet" ]; then \
		docker compose --env-file .env.miner -f docker-compose.miner.yaml up --build -d miner-mainnet-centralised; \
	elif [ "$(network)" = "testnet" ]; then \
		docker compose --env-file .env.miner -f docker-compose.miner.yaml up --build -d miner-testnet-centralised; \
	else \
		echo "Please specify a valid network: mainnet or testnet"; \
	fi


validator:
	@if [ "$(network)" = "mainnet" ]; then \
		docker compose --env-file .env.validator -f docker-compose.validator.yaml up --build -d validator-mainnet; \
	elif [ "$(network)" = "testnet" ]; then \
		docker compose --env-file .env.validator -f docker-compose.validator.yaml up --build -d validator-testnet; \
	else \
		echo "Please specify a valid network: mainnet or testnet"; \
	fi

miner-worker-api:
	docker compose --env-file .env.miner -f docker-compose.miner.yaml up -d worker-api

dojo-cli:
	docker compose --env-file .env.miner -f docker-compose.miner.yaml run --rm dojo-cli

# ---------------------------------------------------------------------------- #
#                             CORE SERVICE LOGGING                             #
# ---------------------------------------------------------------------------- #

miner-decentralised-logs:
	@if [ "$(network)" = "mainnet" ]; then \
		docker compose --env-file .env.miner -f docker-compose.miner.yaml logs -f miner-mainnet-decentralised; \
	elif [ "$(network)" = "testnet" ]; then \
		docker compose --env-file .env.miner -f docker-compose.miner.yaml logs -f miner-testnet-decentralised; \
	else \
		echo "Please specify a valid network: mainnet or testnet"; \
	fi

miner-centralised-logs:
	@if [ "$(network)" = "mainnet" ]; then \
		docker compose --env-file .env.miner -f docker-compose.miner.yaml logs -f miner-mainnet-centralised; \
	elif [ "$(network)" = "testnet" ]; then \
		docker compose --env-file .env.miner -f docker-compose.miner.yaml logs -f miner-testnet-centralised; \
	else \
		echo "Please specify a valid network: mainnet or testnet"; \
	fi

validator-logs:
	@if [ "$(network)" = "mainnet" ]; then \
		docker compose --env-file .env.validator -f docker-compose.validator.yaml logs -f validator-mainnet; \
	elif [ "$(network)" = "testnet" ]; then \
		docker compose --env-file .env.validator -f docker-compose.validator.yaml logs -f validator-testnet; \
	else \
		echo "Please specify a valid network: mainnet or testnet"; \
	fi
