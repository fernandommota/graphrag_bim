from langchain_core.runnables import  RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_ollama import OllamaEmbeddings
import os
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from neo4j import  Driver

from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

from dotenv import load_dotenv

from ifc_to_graph import convert_ifc_to_graph_document


def create_or_connecto_to_graph(database):

    return Neo4jGraph()


def save_graph(entities):

    graph.add_graph_documents(
        entities,
        baseEntityLabel=True,
        include_source=True
    )

graph = create_or_connecto_to_graph("ifc_haus")

graph_ifc_documents = convert_ifc_to_graph_document('input/ifc/AC20-FZK-Haus.ifc')

#for index, graph_document in enumerate(graph_ifc_documents):
#    pass
    #print('index',index)
    #print('graph_document', graph_document)

save_graph(graph_ifc_documents)

def run_query(human_query):
    
    llm = OllamaFunctions(model="llama3.1", temperature=0, format="json",base_url='http://localhost:11434')
    llm_transformer = LLMGraphTransformer(llm=llm)

    driver = GraphDatabase.driver(
            uri = os.environ["NEO4J_URI"],
            auth = (os.environ["NEO4J_USERNAME"],
                    os.environ["NEO4J_PASSWORD"]))

    def create_fulltext_index(tx):
        #query = '''
        #CREATE FULLTEXT INDEX `fulltext_entity_id` 
        #FOR (n:__Entity__) 
        #ON EACH [n.id];
        #'''
        #tx.run(query)

        query = '''
        CREATE FULLTEXT INDEX `fulltext_entity_ifc_door` 
        FOR (n:IfcDoor) 
        ON EACH [n.type];
        '''
        tx.run(query)

        query = '''
        CREATE FULLTEXT INDEX `fulltext_entity_ifc_window` 
        FOR (n:IfcWindow) 
        ON EACH [n.type];
        '''
        tx.run(query)

    ## Function to execute the query
    def create_index():
        with driver.session() as session:
            session.execute_write(create_fulltext_index)
            print("Fulltext index created successfully.")

    # Call the function to create the index
    try:
        create_index()
    except:
        pass

    # Close the driver connection
    driver.close()


    class Entities(BaseModel):
        """Identifying information about entities."""

        names: list[str] = Field(
            ...,
            description="All the person, organization, or business entities that "
            "appear in the text",
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are extracting Industry Foundation Classes (IFC) elements from the text.",
            ),
            (
                "human",
                "Use the given format to extract information from the following "
                "input: {question}",
            ),
        ]
    )


    entity_chain = llm.with_structured_output(Entities)

    #print(f'entity_chain: ',entity_chain.invoke(human_query))


    # Fulltext index query
    def graph_retriever(question: str) -> str:
        """
        Collects the neighborhood of entities mentioned
        in the question
        """
        result = ""
        entities = entity_chain.invoke(question)
        for entity in entities.names:
            #print('entity', entity)
            response = graph.query(
                """CALL db.index.fulltext.queryNodes('fulltext_entity_ifc_door', $query, {limit:200})
                YIELD node,score
                CALL {
                WITH node
                MATCH (node)-[r:IFC]->(neighbor)
                RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                UNION ALL
                WITH node
                MATCH (node)<-[r:IFC]-(neighbor)
                RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": entity},
            )
            result += "\n".join([el['output'] for el in response])

            response = graph.query(
                """CALL db.index.fulltext.queryNodes('fulltext_entity_ifc_window', $query, {limit:200})
                YIELD node,score
                CALL {
                WITH node
                MATCH (node)-[r:IFC]->(neighbor)
                RETURN node.id + ' type ' + node.type + ' name ' + node.name + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                UNION ALL
                WITH node
                MATCH (node)<-[r:IFC]-(neighbor)
                RETURN neighbor.id + ' type ' + node.type + ' name ' + node.name + ' - ' + type(r) + ' -> ' +  node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": entity},
            )
            result += "\n".join([el['output'] for el in response])
        return result

    #print('Prompt graph_retriever',graph_retriever(human_query))

    def full_retriever(question: str):
        graph_data = graph_retriever(question)
        final_data = f"""Graph data:
    {graph_data}
        """
        return final_data

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    Use natural language and be concise.
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
            {
                "context": full_retriever,
                "question": RunnablePassthrough(),
            }
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(input=human_query)

    print(f'Response: {response}')

# Count the number of X IFC Element
# 11 window
# 5 doors

run_query("What is the number of IfcDoor?")
run_query("What is the names of IfcDoor entities and count?")

#run_query("Count the number of unique IfcDoor?")
#run_query("Count the number of unique IfcWindow?")
#run_query("Count the unique number of IIfcWindow appears?")
run_query("What is the number of IfcWindow?")
run_query("What is the names of IfcWindow entities and count?")

# Sum the number of wall