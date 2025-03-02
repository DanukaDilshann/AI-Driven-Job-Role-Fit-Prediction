import pyodbc


server = 'DESKTOP-2DSGQFI'
driver = '{ODBC Driver 17 for SQL Server}'
default_database = 'master'
connection_string = f"DRIVER={driver};SERVER={server};DATABASE={default_database};Trusted_Connection=yes"

def store_Job_description(Job_Role, Required_Programming_Skills, Required_Soft_Skills, Required_Technical_Skills, Professional_Qualifications_with_Years, Required_Educational_Qualifications):
    try:
       
        conn = pyodbc.connect(connection_string, autocommit=True)
        cursor = conn.cursor()
        print("Connected to SQL Server")
        
        
        database_name = 'ABC_Company'
        create_db_query = f"""
        IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = N'{database_name}')
        BEGIN
            CREATE DATABASE [{database_name}]
        END
        """
        cursor.execute(create_db_query)
        print(f"Database '{database_name}' checked/created.")
        cursor.close()
        conn.close()

        
        new_connection_string = f"DRIVER={driver};SERVER={server};DATABASE={database_name};Trusted_Connection=yes"
        conn = pyodbc.connect(new_connection_string, autocommit=True)
        cursor = conn.cursor()

        
        create_table_query = """
        IF OBJECT_ID('Job_Descriptions', 'U') IS NULL
        CREATE TABLE Job_Descriptions (
            Job_Role VARCHAR(200) PRIMARY KEY,
            Required_Programming_Skills VARCHAR(255),
            Required_Soft_Skills VARCHAR(255),
            Required_Technical_Skills VARCHAR(255),
            Professional_Qualifications_with_Years VARCHAR(255),
            Required_Educational_Qualifications VARCHAR(255)
 
        )
        """
        cursor.execute(create_table_query)
        print("Table 'Job_Descriptions' checked/created.")

        
        insert_query = """
        INSERT INTO Job_Descriptions 
        (Job_Role, Required_Programming_Skills, Required_Soft_Skills, Required_Technical_Skills, Professional_Qualifications_with_Years, Required_Educational_Qualifications)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        cursor.execute(insert_query, (
            Job_Role,
            Required_Programming_Skills,
            Required_Soft_Skills,
            Required_Technical_Skills,
            Professional_Qualifications_with_Years,
            Required_Educational_Qualifications
        ))
        print("Data inserted successfully.")

    except pyodbc.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        
        if 'cursor' in locals() and cursor is not None:
            cursor.close()
        if 'conn' in locals() and conn is not None:
            conn.close()
        print("Connection closed")
