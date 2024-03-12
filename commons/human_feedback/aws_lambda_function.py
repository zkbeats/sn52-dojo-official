import os
import json
import boto3
import xml.etree.ElementTree as ET
import traceback
import urllib3

# load environment variables
TARGET_URL = os.environ["TARGET_URL"]
MTURK_ENDPOINT_URL = os.environ["MTURK_ENDPOINT_URL"]


def lambda_handler(event, context):
    http = urllib3.PoolManager()
    try:
        answers = []
        for record in event["Records"]:
            notification = json.loads(record["Sns"]["Message"])

            for mturk_event in notification["Events"]:
                mturk = boto3.client(
                    "mturk", region_name="us-east-1", endpoint_url=MTURK_ENDPOINT_URL
                )

                if mturk_event["EventType"] in ["AssignmentSubmitted"]:
                    # Retrieve the answers that were provided by Workers
                    response = mturk.list_assignments_for_hit(
                        HITId=mturk_event["HITId"]
                    )
                    assignments = response["Assignments"]
                    for assignment in assignments:
                        answers.append(parse_answers(assignment))

                    # Do something with the answers
                    # ...
        encoded_data = json.dumps(answers).encode("utf-8")
        r = http.request(
            method="POST",
            url=TARGET_URL,
            body=encoded_data,
            headers={"Content-Type": "application/json"},
        )

        return {
            "statusCode": r.status,
            "body": json.dumps(answers),
            "forwarded_response": r.data.decode("utf-8"),
        }
    except Exception as e:
        print(f"Encountered exception: {e}")
        traceback.print_exc()
        pass


# Function to parse the Answer XML object
def parse_answers(assignment):
    result = {
        "HITId": assignment["HITId"],
        "WorkerId": assignment["WorkerId"],
        "Answer": [],
    }

    ns = {
        "mt": "http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2005-10-01/QuestionFormAnswers.xsd"
    }
    root = ET.fromstring(assignment["Answer"])

    for a in root.findall("mt:Answer", ns):
        name = a.find("mt:QuestionIdentifier", ns).text
        value = a.find("mt:FreeText", ns).text
        result["Answer"].append({name: json.loads(value)})
    return result
