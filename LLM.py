import os
import requests 

api_key = os.getenv("MIMO_OPENAI_API_KEY")
url = "https://ai.mimo.org/v1/openai/message"
headers = {"api-key": api_key}
current_thread_id = None
threads = []

print("Welcome! Type your message and press Enter to send.")
print("Type 'exit' to end the program.")
print("Type 'new' to switch conversation thread.")
print("Keep in mind that the database is as of Oct 2023. You would need to upload data for analysis.")
print("Starting a new thread for you.\n")

while True:
  user_message = input("You: ")
  if user_message == "exit":
    print("Chat ended")
    break
  elif user_message == "new":
    current_thread_id = None
    print("A new thread is about to start.\n")
    continue 

  def send_message(user_message, current_thread_id):
    body = {
      "message": user_message
    }
    if current_thread_id:
      body["threadId"] = current_thread_id
    response = requests.post(url, headers=headers, json=body)
    data = response.json()
    return data

  response_data = send_message(user_message, current_thread_id)
  latest_message = response_data.get("response")
  current_thread_id = response_data.get("threadId")
  print(f"GPT: {latest_message}")
  if current_thread_id not in threads:
    threads.append(current_thread_id)
