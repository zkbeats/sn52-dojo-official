import argparse
import subprocess
import time

from bittensor.btlogging import logging as logger

from template import __version__

CHECK_INTERVAL = 1800  # 30 minutes

# Define the base URL for the images
BASE_IMAGE_URL = "ghcr.io/tensorplex-labs/"
BRANCH = "main"

CONFIG = {
    "validator": {
        "services": [
            "redis-service",
            "synthetic-api",
            "node-subtensor-testnet",
            "validator-testnet",
        ],
        "images": ["dojo-synthetic-api"],
    },
    "miner": {
        "services": [
            "redis-service",
            "postgres-service",
            "prisma-setup",
            "node-subtensor-testnet",
            "sidecar",
            "worker-api",
            "worker-ui",
            "miner-testnet",
        ],
        "images": ["dojo-worker-api", "dojo-ui"],
    },
}


def get_latest_remote_tag():
    """Fetch and return the latest tag from the remote repository."""
    try:
        result = subprocess.run(
            [
                "git",
                "ls-remote",
                "--tags",
                "https://github.com/tensorplex-labs/dojo.git",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        # Parse the result to get the latest tag
        tags = [line.split("/")[-1] for line in result.stdout.strip().split("\n")]
        latest_tag = sorted(
            tags, key=lambda s: list(map(int, s.strip("v").split(".")))
        )[-1]
        # strip "v"
        latest_tag = latest_tag[1:]
        logger.debug(f"Latest tag from remote: {latest_tag}")
        return latest_tag
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to fetch latest git tag: {e}")
        return None


def get_image_digest(image_name):
    """Get the image digest for the specified Docker image."""
    try:
        digest = (
            subprocess.run(
                [
                    "docker",
                    "inspect",
                    "--format='{{index .RepoDigests 0}}'",
                    image_name,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
            .replace("'", "")
        )
        return digest
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get image digest for {image_name}: {e}")
        return None


def check_for_update(image_url):
    """Check if there is an update available for the Docker image."""
    logger.info(f"Checking for updates for {image_url}...")
    local_digest = get_image_digest(image_url)

    if not local_digest:
        return False

    logger.debug(f"Local digest: {local_digest}")

    # Pull the remote image
    pull_docker_image(image_url)

    remote_digest = get_image_digest(image_url)

    if not remote_digest:
        return False

    logger.debug(f"Remote digest: {remote_digest}")

    if local_digest != remote_digest:
        logger.info(f"Update available for {image_url}.")
        return True
    else:
        logger.info(f"No update available for {image_url}.")
        return False


def pull_docker_image(image_url):
    """Pull the latest Docker image."""
    logger.info(f"Pulling the latest Docker image for {image_url}.")
    try:
        subprocess.run(["docker", "pull", "--quiet", image_url], check=True)
        logger.info(f"Successfully pulled {image_url}.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to pull Docker image {image_url}: {e}")
        return False
    return True


def pull_docker_images(list_of_images: list[str]):
    for image_name in list_of_images:
        image_url = (
            f"{BASE_IMAGE_URL}{image_name}:tensorplex-prod"
            if image_name == "dojo-ui"
            else f"{BASE_IMAGE_URL}{image_name}:{BRANCH}"
        )
        pull_docker_image(image_url)


def check_for_image_updates(images):
    logger.info(f"Checking images: {images}")
    has_update = False
    for image_name in images:
        image_url = (
            f"{BASE_IMAGE_URL}{image_name}:tensorplex-prod"
            if image_name == "dojo-ui"
            else f"{BASE_IMAGE_URL}{image_name}:{BRANCH}"
        )
        result = check_for_update(image_url)
        if result:
            has_update = True
    return has_update


def stash_changes():
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    )
    if result.stdout.strip():
        logger.info("Stashing any local changes.")
        subprocess.run(["git", "stash"], check=True)
    else:
        logger.info("No changes to stash.")


def pull_latest_changes():
    logger.info("Pulling latest changes from the main branch.")
    subprocess.run(["git", "pull", "origin", "main"], check=True)
    subprocess.run(["git", "fetch", "--tags"], check=True)


def pop_stash():
    logger.info("Popping stashed changes.")
    subprocess.run(["git", "stash", "pop"], check=True)


def restart_docker(service_name):
    service_data = CONFIG.get(service_name, {})
    services_to_restart = service_data.get("services", [])
    main_service = "miner-testnet" if service_name == "miner" else "validator-testnet"

    if not services_to_restart:
        logger.error(f"No services found for {service_name}.")
        return

    logger.info(f"Restarting Docker services for: {service_name}.")

    # Stop the services in a single command
    # TODO change based on miner/validator services
    subprocess.run(["docker", "compose", "stop"] + services_to_restart, check=True)

    # Start the services in a single command
    # TODO change based on miner/validator services
    subprocess.run(["docker", "compose", "up", "-d", main_service], check=True)


def get_current_version():
    version = __version__
    logger.debug(f"Current version: {version}")
    return version


def update_repo():
    logger.info("Updating the script and its submodules.")
    stash_changes()
    pull_latest_changes()
    pop_stash()


def main(service_name):
    logger.info("Starting the main loop.")
    config = CONFIG[service_name]

    # Initial update and start the docker services
    current_dojo_version = get_current_version()
    new_dojo_version = get_latest_remote_tag()

    if current_dojo_version != new_dojo_version:
        update_repo()

    # Pull the latest images
    pull_docker_images(config["images"])
    restart_docker(service_name)

    # Start the periodic check loop
    while True:
        logger.info("Checking for updates...")
        has_image_updates = check_for_image_updates(config["images"])

        if current_dojo_version != new_dojo_version:
            logger.info("Repository has changed. Updating...")
            update_repo()
            current_dojo_version = new_dojo_version

        if current_dojo_version != new_dojo_version or has_image_updates:
            restart_docker(service_name)

        logger.info(f"Sleeping for {CHECK_INTERVAL} seconds.")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the auto-update script.")
    parser.add_argument(
        "service",
        choices=["miner", "validator"],
        help="Specify the service to run (miner or validator).",
    )
    args = parser.parse_args()

    logger.info(f"Starting the {args.service} process.")
    main(args.service)
