from model.prepare_data import prepare_data
from model.response import search_db


def chat() -> None:
    """
    Chatbot function that takes user input and responds accordingly.

    :return: None
    """
    db = prepare_data()

    print(
        "Hello! \nI am a chatbot specialized in answering questions about the most popular Dostoevsky's novels. \nYou can ask me anything about the following books: \n\nCrime and Punishment,\nThe Brothers Karamazov,\nThe Idiot,\nWhite Nights,\nOther Stories and Notes from Underground.\n\nIf you want to exit, just type 'bye'.\n"
    )

    while True:
        user_input = input("You: ")
        if user_input.lower() == "bye":
            print("Chatbot: Goodbye! Have a nice day!")
            break
        else:
            response = search_db(db, user_input)
            print(f"Chatbot: {response}")


if __name__ == "__main__":
    chat()
