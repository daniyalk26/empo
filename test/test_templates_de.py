from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

template = """ Sie sind ein Lehrer, der ein Quiz bewertet.
Sie erhalten eine Frage, die Antwort des Schülers und die richtige Antwort und werden gebeten, die Antwort des Schülers 
entweder als RICHTIG oder FALSCH.

Richtiges Beispiel Format:
GRADE: RICHTIG

Korrektes Beispielformat:
GRADE: FALSCH


Benoten Sie die Schülerantworten NUR auf der Grundlage ihrer sachlichen Richtigkeit. Ignorieren Sie Unterschiede in der 
Zeichensetzung und Formulierung zwischen der Schülerantwort und der wahren Antwort. Es ist in Ordnung, wenn die 
Schülerantwort mehr Informationen enthält als die richtige Antwort, solange sie keine widersprüchlichen Aussagen 
enthält.

FRAGE: {query}
STUDENT ANTWORT: {result}
WAHR ANTWORT: {answer}
GRADE:

"""
GRADE_ANSWER_PROMPT = PromptTemplate(input_variables=["query", "result", "answer"], template=template)

generate_qa_system = """

Sie sind ein KI-Assistent, der die Aufgabe hat, Frage-Antwort-Paare aus einem Textstück zu generieren, das aus einem 
einem Dokument über Industrieanlagen. Stellen Sie sicher, dass das Modell Fragen generiert, die detaillierte Antworten erfordern, und nicht 
einfache Ja/Nein-Antworten. Achten Sie darauf, dass Sie benannte Entitäten aus dem bereitgestellten Text einbeziehen. Stellen Sie sicher, dass jede Frage 
den vollständigen Namen der im Titel erwähnten Entität (Gerät) enthalten muss. Achten Sie darauf, allgemeine oder mehrdeutige Fragen zu vermeiden 
Fragen, die sich auf beliebige Abschnitte oder Teile des Dokuments beziehen.

"""

generate_qa_human = """ Bitte denken Sie sich ein Frage/Antwort-Paar aus dem vorgegebenen Text aus. Hier sind die Anweisungen, die Sie
folgen:
Anweisungen:
1. Lesen Sie den vorgegebenen Text aufmerksam durch. Er besteht aus zwei Teilen: einem Titel und einem zufällig ausgewählten Abschnitt aus demselben 
   Dokument.
2. Identifizieren Sie benannte Entitäten im Text. Die im Titel erwähnte Entität muss sich auf das Gerät beziehen, um das es in dem Dokument
 auf das das Dokument ausgerichtet ist.
3. Die benannte Entität in dem bereitgestellten extrahierten Abschnitt könnte sich auf ein Teil/Unterteil der Ausrüstung beziehen, um die es in dem Dokument geht.   
4. Erstellen Sie Fragen zu den Eigenschaften, der Wartung, den Installationsanweisungen usw. der genannten Entitäten. 
5. Stellen Sie sicher, dass die Fragen detaillierte Antworten und keine einfachen Ja/Nein-Antworten erfordern.
6. Achten Sie darauf, generische oder zweideutige Fragen zu vermeiden, die sich auf beliebige Abschnitte oder Teile des Dokuments beziehen.
7. Stellen Sie sicher, dass jede Frage den vollständigen Namen der im Titel genannten Einheit (Gerät) enthält.

Bitte stellen Sie das Frage-Antwort-Paar im folgenden JSON-Format bereit:

```
{{
    "question": "$IHRE_FRAGE_HIER",
    "answer": "$DIE_ANTWORT_HIER"
}}
```

Alles zwischen dem ```` muss gültiges json sein.
Hier ist der bereitgestellte Text:
----------------
{text}
"""

CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(generate_qa_system),
        HumanMessagePromptTemplate.from_template(generate_qa_human),
    ]
)

extract_titles_template = """Ein Auszug aus dem ersten Abschnitt des Dokuments wird Ihnen unten gegeben.
Sie sollen uns den optimalen Titel dieses Dokuments nennen, einschließlich der benannten Entitäten, sofern sie im Text vorhanden sind.


Extrahierter Text: {text}

Kurze Beschreibung des Dokuments:"""
DOCUMENT_DESCRIPTION_PROMPT = PromptTemplate(template=extract_titles_template, input_variables=["text"])

filter_questions_template = """
Ein Textstück wird Ihnen zur Verfügung gestellt, das aus zwei Teilen besteht: Der erste Teil ist ein Frage-Antwort-Paar, 
das aus einem Dokument generiert wurde, und der zweite Teil ist der Titel des Dokuments, aus dem dieses Frage-Antwort-Paar extrahiert wurde. 
Im Titel des Dokuments befindet sich der Name einer Ausrüstung. Sie müssen die Frage lesen und überprüfen, ob die im 
Titel genannte benannte Entität auch in der Frage erwähnt wird oder nicht. Generieren Sie eine binäre Antwort 0 oder 1. 
Generieren Sie '0', wenn die benannte Entität aus dem Titel nicht in der Frage vorhanden ist. Generieren Sie '1', 
wenn die benannte Entität aus dem Titel in der Frage vorhanden ist.

Bereitgestellter Text: {text}
Antwort: 
"""
QUESTION_FILTER_PROMPT = PromptTemplate(template=filter_questions_template, input_variables=["text"])


