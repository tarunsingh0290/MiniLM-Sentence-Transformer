
from time import perf_counter

from sentence_transformers import SentenceTransformer, util


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.70


# Sample existing support tickets
existing_tickets = [
    {
        "ticket_id": "T101",
        "description": "I was charged twice for my monthly subscription."
    },
    {
        "ticket_id": "T102",
        "description": "I cannot log in to my account after resetting my password."
    },
    {
        "ticket_id": "T103",
        "description": "My invoice shows an incorrect amount."
    },
    {
        "ticket_id": "T104",
        "description": "The application crashes whenever I open the dashboard."
    },
    {
        "ticket_id": "T105",
        "description": "I was billed two times for the same subscription."
    },
]


# New ticket to compare against existing tickets
new_ticket = (
    "it shows that my account is deleted"
)


def main():
    print("=" * 60)
    print("MINILM SEMANTIC DUPLICATE DETECTION TEST")
    print("=" * 60)

    # 1. Load the model on CPU
    print("\nLoading MiniLM model...")

    load_start = perf_counter()

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    load_time = perf_counter() - load_start

    print(f"Model loaded successfully in {load_time:.2f} seconds")

    # 2. Extract ticket IDs and descriptions
    ticket_ids = [
        ticket["ticket_id"]
        for ticket in existing_tickets
    ]

    descriptions = [
        ticket["description"]
        for ticket in existing_tickets
    ]

    # 3. Generate embeddings
    print("\nGenerating ticket embeddings...")

    embedding_start = perf_counter()

    existing_embeddings = model.encode(
        descriptions,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    new_embedding = model.encode(
        [new_ticket],
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embedding_time = perf_counter() - embedding_start

    print(f"Existing tickets: {len(descriptions)}")
    print(f"Embedding dimensions: {existing_embeddings.shape[1]}")
    print(f"Embedding generation time: {embedding_time:.3f} seconds")

    # 4. Calculate cosine similarity
    similarity_start = perf_counter()

    scores = util.cos_sim(
        new_embedding,
        existing_embeddings
    )[0]

    similarity_time = perf_counter() - similarity_start

    # 5. Rank tickets by similarity
    ranked_results = sorted(
        zip(ticket_ids, descriptions, scores.tolist()),
        key=lambda result: result[2],
        reverse=True
    )

    print("\nNEW TICKET")
    print("-" * 60)
    print(new_ticket)

    print("\nRANKED SIMILARITY RESULTS")
    print("-" * 60)

    for rank, (ticket_id, description, score) in enumerate(
        ranked_results,
        start=1
    ):
        if score >= SIMILARITY_THRESHOLD:
            result = "POTENTIAL MATCH"
        else:
            result = "BELOW THRESHOLD"

        print(f"\nRank: {rank}")
        print(f"Ticket ID: {ticket_id}")
        print(f"Description: {description}")
        print(f"Cosine similarity: {score:.4f}")
        print(f"Result: {result}")

    # 6. Performance summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)

    print(f"Model: {MODEL_NAME}")
    print("Device: CPU")
    print(f"Model loading time: {load_time:.2f} seconds")
    print(f"Embedding generation time: {embedding_time:.3f} seconds")
    print(f"Similarity calculation time: {similarity_time:.6f} seconds")
    print(f"Existing tickets compared: {len(existing_tickets)}")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")

    print("\nTest completed successfully.")


if __name__ == "__main__":
    main()
