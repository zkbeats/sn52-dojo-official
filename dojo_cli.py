import argparse
from typing import Callable, Dict
import bittensor
import requests
from rich.console import Console
from prompt_toolkit import PromptSession, prompt
from prompt_toolkit.completion import WordCompleter, FuzzyCompleter

console = Console()

DOJO_API_BASE_URL = "http://localhost:9946"


def success(message: str, emoji: str = ":white_check_mark:"):
    console.print(f"{emoji} [green]{message}[/green]")


def info(message: str, emoji: str = ":information_source:"):
    console.print(f"{emoji} [white]{message}[/white]")


def error(message: str, emoji: str = ":x:"):
    console.print(f"{emoji} [red]{message}[/red]")


def warning(message: str, emoji: str = ":warning:"):
    console.print(f"{emoji} [yellow]{message}[/yellow]")


def api_key_list(cookies):
    response = requests.get(
        f"{DOJO_API_BASE_URL}/api/v1/miner/api-key/list", cookies=cookies
    )
    response.raise_for_status()
    keys = response.json().get("body", {}).get("apiKeys")
    if len(keys) == 0:
        warning("No API keys found, please generate one.")
    else:
        success(f"All API keys: {keys}")
    return keys


def api_key_generate(cookies):
    response = requests.post(
        f"{DOJO_API_BASE_URL}/api/v1/miner/api-key/generate", cookies=cookies
    )
    response.raise_for_status()
    keys = response.json().get("body", {}).get("apiKeys")
    success(f"All API keys: {keys}")
    return keys


def api_key_delete(cookies):
    keys = api_key_list(cookies)
    if not keys:
        return

    key_completer = FuzzyCompleter(WordCompleter(keys, ignore_case=True))
    selected_key = prompt(
        "Select an API key to delete: ",
        completer=key_completer,
        swap_light_and_dark_colors=True,
    )
    if selected_key not in keys:
        error("Invalid selection.")
        return

    response = requests.put(
        f"{DOJO_API_BASE_URL}/api/v1/miner/api-key/disable",
        json={"apiKey": selected_key},
        cookies=cookies,
    )
    response.raise_for_status()
    remaining_keys = response.json().get("body", {}).get("apiKeys")
    success(f"Remaining API keys: {remaining_keys}")
    return


def _get_session_cookies(hotkey: str, signature: str, message: str):
    url = f"{DOJO_API_BASE_URL}/api/v1/miner/session/auth"
    if not signature.startswith("0x"):
        signature = "0x" + signature

    payload = {
        "hotkey": hotkey,
        "signature": signature,
        "message": message,
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    cookies = response.cookies.get_dict()
    return cookies


class State:
    def __init__(self, config):
        self.cookies = None
        self.wallet = bittensor.wallet(config=config)
        # cli = bittensor.cli(config=config)
        # self.subtensor = bittensor.subtensor(config=config)
        # axons = self.subtensor.metagraph(netuid=cli.config.netuid, lite=False).axons


def get_session_cookies(wallet):
    info("Grabbing wallet hotkey from keypair file... please input your password")
    wallet.coldkey

    kp = wallet.hotkey
    hotkey = str(wallet.hotkey.ss58_address)

    raw_message = "Sign in to Dojo with Substrate"

    def prepare_message(message: str):
        return f"<Bytes>{message}</Bytes>"

    prepared_message = prepare_message(raw_message)
    signature = kp.sign(prepared_message).hex()
    try:
        cookies = _get_session_cookies(hotkey, signature, raw_message)
        success("Successfully got session cookies :cookie:")
        return cookies
    except Exception as e:
        error(f"Failed to get session cookies due to exception: {e}")
        pass
    return


def placeholder():
    success("go implement it")


# TODO
subscription_key_actions = {
    "list": placeholder,
    "generate": placeholder,
    "delete": placeholder,
}

api_key_actions = {
    "list": api_key_list,
    "generate": api_key_generate,
    "delete": api_key_delete,
}

nested_actions: Dict[str, Callable] = {
    "authenticate": get_session_cookies,
    "api_key": api_key_actions,
    "subscription_key": subscription_key_actions,
}


def nested_dict_none(d):
    """replace values which are func calls with None else assertion error"""
    if not isinstance(d, dict):
        return None
    return {k: nested_dict_none(v) for k, v in d.items()}


# nested_completer_data = nested_dict_none(nested_actions)


def flatten_nested_dict(d, parent_key="", sep=" "):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def main():
    parser = argparse.ArgumentParser(description="Bittensor Wallet CLI")
    bittensor.wallet.add_args(parser)
    config = bittensor.config(parser)
    info(f"Using bittensor config:\n{config}")

    is_wallet_set = False
    while not is_wallet_set:
        if config.wallet.name == "default" or config.wallet.hotkey == "default":
            info("Please specify the wallet name and hotkey.")
            config.wallet.name = input("Enter the wallet name: ").strip()
            config.wallet.hotkey = input("Enter the wallet hotkey: ").strip()
        else:
            is_wallet_set = True

    success("Wallet name and hotkey set successfully.")

    state = State(config)
    # method_completer = NestedCompleter.from_nested_dict(nested_dict_none(actions))
    flattened_actions = flatten_nested_dict(nested_actions)
    method_completer = WordCompleter(words=flattened_actions.keys(), ignore_case=True)
    session = PromptSession(
        completer=method_completer, swap_light_and_dark_colors=False
    )

    while True:
        try:
            text = session.prompt(">>> ", completer=method_completer)
            parsed_text = text.strip().lower()
            if parsed_text == "exit":
                break

            action = flattened_actions.get(parsed_text)
            if action:
                if action == get_session_cookies:
                    state.cookies = action(state.wallet)
                else:
                    if state.cookies is None:
                        warning("No session found, please authenticate first.")
                        continue
                    action(state.cookies)
            else:
                warning("Invalid action, please try again")
        except (KeyboardInterrupt, EOFError):
            break
        except Exception:
            pass


if __name__ == "__main__":
    main()
