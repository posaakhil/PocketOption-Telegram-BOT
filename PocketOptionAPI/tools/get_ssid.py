import os
import json
import time
import re
import logging
from typing import cast, List, Dict, Any
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from driver import get_driver

# Configure logging for this script to provide clear, structured output.
# Logs will be directed to standard output, making them compatible with containerization
# and centralized log collection systems.
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


def save_to_env(key: str, value: str):
    """
    Saves or updates a key-value pair in the .env file.
    If the key already exists, its value is updated. Otherwise, the new key-value pair is added.

    Args:
        key: The environment variable key (e.g., "SSID").
        value: The value to be associated with the key.
    """
    env_path = os.path.join(os.getcwd(), ".env")
    lines = []
    found = False

    # Read existing .env file content
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if line.strip().startswith(f"{key}="):
                    # Update existing key
                    lines.append(f'{key}="{value}"\n')
                    found = True
                else:
                    lines.append(line)

    if not found:
        # Add new key if not found
        lines.append(f'{key}="{value}"\n')

    # Write updated content back to .env file
    with open(env_path, "w") as f:
        f.writelines(lines)
    logger.info(f"Successfully saved {key} to .env file.")


def get_pocketoption_ssid():
    """
    Automates the process of logging into PocketOption, navigating to a specific cabinet page,
    and then scraping WebSocket traffic to extract the session ID (SSID).
    The extracted SSID is then saved to the .env file.
    """
    driver = None
    try:
        # Initialize the Selenium WebDriver using the helper function from driver.py.
        # This ensures the browser profile is persistent for easier logins.
        driver = get_driver("chrome")
        login_url = "https://pocketoption.com/en/login"
        cabinet_base_url = "https://pocketoption.com/en/cabinet"
        target_cabinet_url = "https://pocketoption.com/en/cabinet/demo-quick-high-low/"
        # Regex to capture the entire "42[\"auth\",{...}]" string.
        # This pattern is designed to be robust and capture the full authentication message,
        # regardless of the specific content of the 'session' field (e.g., simple string or serialized PHP array).
        ssid_pattern = r'(42\["auth",\{"session":"[^"]+","isDemo":\d+,"uid":\d+,"platform":\d+,"isFastHistory":(?:true|false)\}\])'

        logger.info(f"Navigating to login page: {login_url}")
        driver.get(login_url)

        # Wait indefinitely for the user to manually log in and be redirected to the cabinet base page.
        # This uses an explicit wait condition to check if the current URL contains the cabinet_base_url.
        logger.info(f"Waiting for user to login and redirect to {cabinet_base_url}...")
        WebDriverWait(driver, 9999).until(EC.url_contains(cabinet_base_url))
        logger.info("Login successful. Redirected to cabinet base page.")

        # Now navigate to the specific target URL within the cabinet.
        logger.info(f"Navigating to target cabinet page: {target_cabinet_url}")
        driver.get(target_cabinet_url)

        # Wait for the target cabinet URL to be fully loaded.
        # This ensures that any WebSocket connections initiated on this page are established.
        WebDriverWait(driver, 60).until(EC.url_contains(target_cabinet_url))
        logger.info("Successfully navigated to the target cabinet page.")

        # Give the page some time to load all WebSocket connections and messages after redirection.
        # This delay helps ensure that the relevant WebSocket frames are captured in the logs.
        time.sleep(5)

        # Retrieve performance logs which include network requests and WebSocket frames.
        # These logs are crucial for capturing the raw WebSocket messages.
        get_log = getattr(driver, "get_log", None)
        if not callable(get_log):
            raise AttributeError(
                "Your WebDriver does not support get_log(). Make sure you are using Chrome with performance logging enabled."
            )
        performance_logs = cast(List[Dict[str, Any]], get_log("performance"))
        logger.info(f"Collected {len(performance_logs)} performance log entries.")

        found_full_ssid_string = None
        # Iterate through the performance logs to find WebSocket frames.
        for entry in performance_logs:
            message = json.loads(entry["message"])
            # Check if the log entry is a WebSocket frame (either sent or received)
            # and contains the desired payload data.
            if (
                message["message"]["method"] == "Network.webSocketFrameReceived"
                or message["message"]["method"] == "Network.webSocketFrameSent"
            ):
                payload_data = message["message"]["params"]["response"]["payloadData"]
                # Attempt to find the full SSID string using the defined regex pattern.
                match = re.search(ssid_pattern, payload_data)
                if match:
                    # Capture the entire matched group as the full SSID string.
                    found_full_ssid_string = match.group(1)
                    logger.info(
                        f"Found full SSID string in WebSocket payload: {found_full_ssid_string}"
                    )
                    # Break after finding the first match as it's likely the correct one.
                    break

        if found_full_ssid_string:
            # Save the extracted full SSID string to the .env file.
            save_to_env("SSID", found_full_ssid_string)
            logger.info("Full SSID string successfully extracted and saved to .env.")
        else:
            logger.warning(
                "Full SSID string pattern not found in WebSocket logs after login."
            )

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        # Ensure the WebDriver is closed even if an error occurs to free up resources.
        if driver:
            driver.quit()
            logger.info("WebDriver closed.")


if __name__ == "__main__":
    get_pocketoption_ssid()
