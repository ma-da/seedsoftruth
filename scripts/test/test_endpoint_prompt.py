import model_controller

def main():
    prompt = "Who killed JFK?"
    max_new_tokens = 500
    response = model_context.send_prompt_to_endpoint(prompt, max_new_tokens)
    print(f"Model response: {response}")

if __name__ == "__main__":
    main()