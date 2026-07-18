#----------------------------------------------------------------------------------------------
#                                       Import Statements
#----------------------------------------------------------------------------------------------

import psycopg2
from dotenv import load_dotenv
import os

#----------------------------------------------------------------------------------------------
#                                       Init Statements
#----------------------------------------------------------------------------------------------
# Load environment variables from .env
load_dotenv()
# Fetch database URL
DATABASE_URL = os.getenv("DATABASE_URL")

#----------------------------------------------------------------------------------------------
#                                       Logic Statements
#----------------------------------------------------------------------------------------------

def create_table():
    """
        Logic of creating table
    """
    # Connect to Supabase PostgreSQL
try:
    connection = psycopg2.connect(
        DATABASE_URL,
        sslmode="require"
    )

    print("Connection Successful")

    cursor = connection.cursor()

    # Create table
    create_table_query = """
    CREATE TABLE IF NOT EXISTS products (
        ProductID INT PRIMARY KEY,
        ProductName VARCHAR(255),
        ProductBrand VARCHAR(255),
        Gender VARCHAR(50),
        Price VARCHAR(50),
        Description TEXT,
        PrimaryColor VARCHAR(50)
    );
    """

    cursor.execute(create_table_query)
    print("Table created successfully")

    # Save changes
    connection.commit()


except Exception as e:
    print("Error in creating table :- " , str(e))


create_table()