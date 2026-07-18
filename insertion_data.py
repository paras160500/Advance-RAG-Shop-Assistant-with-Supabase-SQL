#----------------------------------------------------------------------------------------------
#                                       Import Statements
#----------------------------------------------------------------------------------------------

import psycopg2
from dotenv import load_dotenv
import os
import pandas as pd 

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

try:
    # Getting connection
    connection = psycopg2.connect(
        DATABASE_URL,
        sslmode="require"
    )

    print("Connection Successful")
    # Getting cursor from connection
    cursor = connection.cursor()

    csv_file = "data.csv"
    try:
        # Check whether the table already contains data
        cursor.execute("SELECT COUNT(*) FROM products")
        row_count = cursor.fetchone()[0]

        if row_count > 0:
            print(f"Table already contains {row_count} rows.")
            print("No data was inserted.")

        else:
            # Reading the CSV file
            data = pd.read_csv(csv_file)
            # Query Generation for SQL push
            sql = """
                INSERT INTO products
                (ProductID, ProductName, ProductBrand, Gender, Price, Description, PrimaryColor)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            # Iterate over all the data and then push it to db
            for _, row in data.iterrows():
                cursor.execute(sql, tuple(row))

            connection.commit()
            print("All data inserted successfully.")

    except Exception as e:
        # Doing Rollback
        connection.rollback()   
        print("Error:", e)

    finally:
        # Final Commit
        cursor.close()
        connection.close()

    
except Exception as e:
    print("Error :- " , str(e))



# ----------------------------For someone who want to insert thousand rows---------------------------------------------
# Convert DataFrame rows to list of tuples
# values = [tuple(row) for _, row in data.iterrows()]

# Bulk insert
# execute_values(cursor, sql, values)