import pyodbc

# Connection details
server = 'DESKTOP-1LBAH96'
driver = '{ODBC Driver 17 for SQL Server}'
default_database = 'master'  # Default database for creating a new database
connection_string = f"DRIVER={driver};SERVER={server};DATABASE={default_database};Trusted_Connection=yes"

def store_similarity_score(emp_id, similarity_measure):
    try:
        # Connect to SQL Server (default database)
        conn = pyodbc.connect(connection_string, autocommit=True)
        cursor = conn.cursor()
        print("Connected to SQL Server")

        # Create the Employee_Data database if it doesn't exist
        database_name = 'Employee_Data'
        create_db_query = f"""
        IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = N'{database_name}')
        BEGIN
            CREATE DATABASE {database_name}
        END
        """
        cursor.execute(create_db_query)
        print(f"Database '{database_name}' checked/created.")

        # Close the connection to master and reconnect to the new database
        cursor.close()
        conn.close()

        # Reconnect to the newly created database
        new_connection_string = f"DRIVER={driver};SERVER={server};DATABASE={database_name};Trusted_Connection=yes"
        conn = pyodbc.connect(new_connection_string, autocommit=True)
        cursor = conn.cursor()

        # Create the Employee table if it doesn't exist
        create_table_query = """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Employee' AND xtype='U')
        BEGIN
            CREATE TABLE Employee (
                Emp_id INT PRIMARY KEY,
                similarity_measure FLOAT
            )
        END
        """
        cursor.execute(create_table_query)
        print("Table 'Employee' checked/created.")

        # Insert data into the Employee table
        insert_query = """
        INSERT INTO Employee (Emp_id, similarity_measure)
        VALUES (?, ?)
        """
        cursor.execute(insert_query, (emp_id, similarity_measure))
        print("Data inserted successfully.")
    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Close the connection
        if 'cursor' in locals() and cursor is not None:
            cursor.close()
        if 'conn' in locals() and conn is not None:
            conn.close()
        print("Connection closed")
