from unittest.mock import MagicMock, patch
import pytest
from src.rag.chatbot import ask_question

@patch("src.rag.chatbot.get_rag_chain")
def test_ask_question_auto_language(mock_get_chain):
    # Mock chain
    mock_chain = MagicMock()
    mock_get_chain.return_value = mock_chain
    mock_chain.invoke.return_value = {"answer": "This is a test answer."}

    # Test Auto language
    question = "Hello"
    answer = ask_question(question, language="Auto")
    
    assert answer == "This is a test answer."
    mock_chain.invoke.assert_called_with({"input": question, "language": "the same language as the question"})

@patch("src.rag.chatbot.get_rag_chain")
def test_ask_question_specific_language(mock_get_chain):
    # Mock chain
    mock_chain = MagicMock()
    mock_get_chain.return_value = mock_chain
    mock_chain.invoke.return_value = {"answer": "Respuesta de prueba."}

    # Test Spanish language
    question = "Hola"
    answer = ask_question(question, language="Spanish")
    
    assert answer == "Respuesta de prueba."
    mock_chain.invoke.assert_called_with({"input": question, "language": "Spanish"})
