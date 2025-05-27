from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import torch.nn.functional as F

# Define paths
MODEL_NAME = "./t5_topic_model/checkpoint-91485"  # Path to the saved checkpoint
MODEL_NAME = "M:/Personal/SE/bachelors/python/test/t5_topic_model/checkpoint-96300"  # Path to the saved checkpoint
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Ensure model is on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def split_into_chunks(text, max_length=256):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]


def generate_topics(article):
    chunks = split_into_chunks(article)
    results = []

    for chunk in chunks:
        input_ids = torch.tensor([chunk]).to(device)
        output_ids = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)

        # Get logits for the generated tokens
        with torch.no_grad():
            outputs = model(input_ids=input_ids, decoder_input_ids=output_ids[:, :-1])
            logits = outputs.logits

        # Calculate log-probabilities and confidence
        probs = F.log_softmax(logits, dim=-1)
        selected_probs = probs.gather(2, output_ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # Shape: (1, seq_len)
        avg_log_prob = selected_probs.mean().item()
        confidence_score = torch.exp(selected_probs.mean()).item()  # convert log prob to prob

        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        results.append((decoded, confidence_score))

    # Print and return unique topics with confidence
    seen = set()
    unique_results = []
    for topic, score in results:
        if topic not in seen:
            seen.add(topic)
            unique_results.append(f"{topic} (confidence: {score:.4f})")

    return "\n".join(unique_results)

# Example usage
articles = []

for i, article in enumerate(articles, 1):
    predicted_topics = generate_topics(article)
    print(f"Article {i} predicted topics:", predicted_topics)
