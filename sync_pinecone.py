#----------------------------------------------------------------------------------------------
#                                       Import Statements
#----------------------------------------------------------------------------------------------
import os 
import time 
import psycopg2
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import pandas as pd 
from tqdm import tqdm

#----------------------------------------------------------------------------------------------
#                                       Init Statements
#----------------------------------------------------------------------------------------------
# Load environment variables from .env
load_dotenv()
# Fetch database URL
DATABASE_URL = os.getenv("DATABASE_URL")
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)
spec = ServerlessSpec(
    cloud = "aws" , region = "us-east-1"
)
index_name = "shop-product-sample"

# Check if index is already existst or not
existing_index = [index_info['name'] for index_info in pc.list_indexes()]
if index_name not in existing_index:
    pc.create_index(
        name = index_name ,
        dimension=1536,
        spec = spec,
        metric = 'dotproduct'
    )
    print("Waiting for Pinecone to init the index...")
    time.sleep(8)
    print("Index Created...")

# connect to index
index = pc.Index(index_name)

try:
    # Getting connection
    connection = psycopg2.connect(
        DATABASE_URL,
        sslmode="require"
    )

    print("Connection Successful")
    # Getting cursor from connection
    cursor = connection.cursor()
except Exception as e:
    print("Error in the connection in sync_pinecone :- " , str(e))

# Embedding model
embed_model = OpenAIEmbeddings(
    api_key=os.getenv("OPEN_AI_API"),
    model="text-embedding-3-small"
)

def fetch_data():
    query = "SELECT * FROM products"
    try:
        cursor.execute(query=query)
        columns = [desc[0] for desc in cursor.description]
        data = pd.DataFrame(cursor.fetchall() , columns=columns)
        print(data.columns.tolist())
        return data 
    except Exception as e:
        print("Error in fetch data :- " , str(e))


def sync_with_pinecone(data):
    batch_size = 100 
    total_batches = (len(data) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0 , len(data) , batch_size), desc="Processing Batches" , unit="batch" , total=total_batches):
        i_end = min(len(data) , i + batch_size)
        batch = data.iloc[i : i_end]

        # Unique Id 
        ids = [str(row['productid']) for _,row in batch.iterrows()]

        # Combine text field for embedding
        texts = [
            f"{row['description']} {row['productname']} {row['productbrand']} {row['gender']} {row['price']} {row['primarycolor']}"
            for _,row in batch.iterrows()
        ]    

        # Embed Text
        embeds = embed_model.embed_documents(texts)

        # Get metadata
        metadata = [
            {
                'ProductName' : row['productname'],
                'ProductBrand' : row['productbrand'],
                'Gender' : row['gender'],
                'Price' : row['price'],
                'PrimaryColor' : row['primarycolor'],
                'Description' : row['description']
            }
            for _ , row in batch.iterrows()
        ]

        # Upsert Vectors
        with tqdm(total = len(ids) , desc="Upserting vectors" , unit = "vector") as upsert_vector:
            index.upsert(vectors=zip(ids , embeds , metadata))
            upsert_vector.update(len(ids))


def main():
    data = fetch_data()
    sync_with_pinecone(data) 


if __name__ == "__main__":
    main()


cursor.close()
connection.close()