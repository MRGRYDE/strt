import os
import tkinter as tk
from tkinter import Scrollbar, Toplevel, messagebox, simpledialog, filedialog
from tkinter import ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import sqlite3
import json

# Global variables for dataset and processing
df = None
X_scaled = None
X = None
y = None
scaler = None

# Database connection functions for CRUD operations
def connect_db(db_name):
    try:
        conn = sqlite3.connect(db_name)
        print(f"Connected to database {db_name}")
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return None

# Create users table
def create_table(conn):
    try:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                features TEXT NOT NULL
            )
        ''')
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error creating table: {e}")

# Create (Insert new user with name and features)
def create_user(conn, name, features, user_id):
    try:
        cursor = conn.cursor()
        features_json = json.dumps(features)
        cursor.execute('''
            INSERT INTO users (id, name, features) VALUES (?, ?, ?)
        ''', (user_id, name, features_json))
        conn.commit()
        messagebox.showinfo("Success", f"User '{name}' added successfully.")
    except sqlite3.Error as e:
        messagebox.showerror("Error", f"Failed to add user: {e}")


# Read (Get all users)
def get_all_users(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users")
        rows = cursor.fetchall()
        return rows
    except sqlite3.Error as e:
        print(f"Error fetching data: {e}")
        return []

# Update (Update user features)
def update_user_features(conn, user_id, new_features):
    try:
        cursor = conn.cursor()
        new_features_json = json.dumps(new_features)
        cursor.execute('''
            UPDATE users SET features = ? WHERE id = ?
        ''', (new_features_json, user_id))
        conn.commit()
        messagebox.showinfo("Success", f"User ID {user_id}'s features updated.")
    except sqlite3.Error as e:
        messagebox.showerror("Error", f"Failed to update user: {e}")

# Delete (Delete user by ID)
def delete_user(conn, user_id):
    try:
        cursor = conn.cursor()
        cursor.execute('''
            DELETE FROM users WHERE id = ?
        ''', (user_id,))
        conn.commit()
        messagebox.showinfo("Success", f"User ID {user_id} deleted successfully.")
    except sqlite3.Error as e:
        messagebox.showerror("Error", f"Failed to delete user: {e}")

# Create CRUD window for managing storage
def storage_window():
    conn = connect_db('heart_patient.db')
    if conn:
        create_table(conn)

    def display_all_users():
        rows = get_all_users(conn)
        display_text = ""
        for row in rows:
            user_id, name, features_json = row
            features = json.loads(features_json)
            display_text += f"ID: {user_id}, Name: {name}, Features: {features}\n"
        if not display_text:
            messagebox.showinfo("No Data", "No users found.")
        else:
            create_output_window("All Users", display_text)

    def add_user():
        name = simpledialog.askstring("Input", "Enter user name:")
        if name:
             user_id_input = simpledialog.askstring("Input", "Enter user ID:")
             if user_id_input:
                 try:
                     user_id = int(user_id_input)
                     rows = get_all_users(conn)
                     for row in rows:
                         if row[0] == user_id:
                             messagebox.showerror("Error", "User ID already in use.")
                             return
                     features_input = simpledialog.askstring("Input", "Enter feature values separated by commas (e.g. 1,2,3,4):")
                     if features_input:
                         try:
                             features = [float(x.strip()) for x in features_input.split(',')]
                             create_user(conn, name, features, user_id)
                         except ValueError:
                             messagebox.showerror("Invalid Input", "Please enter a valid list of numbers separated by commas.")
                     else:
                         messagebox.showerror("Input Error", "No features entered.")
                 except ValueError:
                     messagebox.showerror("Invalid Input", "Please enter a valid integer for user ID.")
             else:
                    messagebox.showerror("Input Error", "No user ID entered.")
        else:
            messagebox.showerror("Input Error", "No name entered.")

        name = simpledialog.askstring("Input", "Enter user name:")
        if name:
            features_input = simpledialog.askstring("Input", "Enter feature values separated by commas (e.g. 1,2,3,4):")
            if features_input:
                try:
                    features = [float(x.strip()) for x in features_input.split(',')]
                    create_user(conn, name, features)
                except ValueError:
                    messagebox.showerror("Invalid Input", "Please enter a valid list of numbers separated by commas.")
            else:
                messagebox.showerror("Input Error", "No features entered.")
        else:
            messagebox.showerror("Input Error", "No name entered.")

    def update_user():
        user_id = simpledialog.askinteger("Input", "Enter user ID to update:")
        if user_id:
            new_features_input = simpledialog.askstring("Input", "Enter new feature values separated by commas (e.g. 1,2,3,4):")
            if new_features_input:
                try:
                    new_features = [float(x.strip()) for x in new_features_input.split(',')]
                    update_user_features(conn, user_id, new_features)
                except ValueError:
                    messagebox.showerror("Invalid Input", "Please enter a valid list of numbers separated by commas.")
            else:
                 messagebox.showerror("Input Error", "No new features entered.")
        else:   
              messagebox.showerror("Input Error", "No user ID entered.")


    def delete_user_action():
        user_id = simpledialog.askinteger("Input", "Enter user ID to delete:")
        if user_id:
            rows = get_all_users(conn)
            if not rows:
                messagebox.showerror("Error", "No users found in the database.")
            else:
                delete_user(conn, user_id)
        else:
                 messagebox.showerror("Input Error", "No user ID entered.")

    # Create a new top-level window for CRUD operations
    storage_win = Toplevel()
    storage_win.title("Storage Operations")
    storage_win.geometry("400x400")

    # Create buttons for CRUD operations
    add_button = tk.Button(storage_win, text="Add User", command=add_user, width=20)
    add_button.pack(pady=10)

    display_button = tk.Button(storage_win, text="Display All Users", command=display_all_users, width=20)
    display_button.pack(pady=10)

    update_button = tk.Button(storage_win, text="Update User", command=update_user, width=20)
    update_button.pack(pady=10)

    delete_button = tk.Button(storage_win, text="Delete User", command=delete_user_action, width=20)
    delete_button.pack(pady=10)

# GUI Functions for displaying dataset information
def create_output_window(title, content):
    window = Toplevel()
    window.title(title)
    window.geometry("800x600")
    text_area = tk.Text(window, wrap="word", font=("Arial", 12))  # Fixed line
    text_area.insert("1.0", content)
    text_area.config(state="disabled")  # Make text read-only
    text_area.pack(expand=True, fill="both")
    scrollbar = Scrollbar(window, command=text_area.yview)
    text_area["yscrollcommand"] = scrollbar.set
    scrollbar.pack(side="right", fill="y")

def create_table_window(title, dataframe):
    window = Toplevel()
    window.title(title)
    window.geometry("800x400")

    # Create Treeview
    tree = ttk.Treeview(window)
    tree.pack(expand=True, fill="both")

    # Define columns
    tree["columns"] = list(dataframe.columns)
    tree["show"] = "headings"

    # Set column headings
    for col in dataframe.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100, anchor="center")

    # Insert rows
    for _, row in dataframe.iterrows():
        tree.insert("", "end", values=list(row))

    # Add a scrollbar
    scrollbar = Scrollbar(window, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")

# Load dataset function
def load_dataset():
    global df, X, y, X_scaled, scaler

    # Open file dialog to select CSV file
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        try:
            df = pd.read_csv(file_path)
            messagebox.showinfo("Success", f"Dataset loaded from {os.path.basename(file_path)}")
            #messagebox.showinfo("Let’s take a look at your reports to better understand what's going on..")
            messagebox.showinfo("    +     ",f"let’s take a look at your reports to better understand what's going on..")
            # Prepare data for scaling and modeling
            X = df.drop('target', axis=1)
            y = df['target']
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")

# GUI Functions for displaying dataset information
def display_shape():
    if df is not None:
        messagebox.showinfo("Dataset Shape", f"Shape of the dataset: {df.shape}")
    else:
        messagebox.showerror("Error", "No dataset loaded")

def display_column_headers():
    if df is not None:
        # Create a DataFrame for column headers
        headers_df = pd.DataFrame({'Index': range(1, len(df.columns) + 1), 'Column Name': df.columns})
        
        # Use the table window to display column headers
        create_table_window("Column Headers", headers_df)
    else:
        messagebox.showerror("Error", "No dataset loaded")

def display_data_types():
    if df is not None:
        # Create a DataFrame for data types
        data_types_df = pd.DataFrame({'Column': df.columns, 'Data Type': df.dtypes}).reset_index(drop=True)
        
        # Use the table window to display data types
        create_table_window("Data Types", data_types_df)
    else:
        messagebox.showerror("Error", "No dataset loaded")

def display_head():
    if df is not None:
        create_table_window("First 5 Rows of Dataset", df.head())
    else:
        messagebox.showerror("Error", "No dataset loaded")

def display_tail():
    if df is not None:
         create_table_window("Last 5 Rows of Dataset", df.tail())
    else:
        messagebox.showerror("Error", "No dataset loaded")

def check_null_values():
    if df is not None:
        # Create a DataFrame for null values
        nulls_df = pd.DataFrame({'Column': df.columns, 'Null Values': df.isnull().sum()}).reset_index(drop=True)
        
        # Use the table window to display null values
        create_table_window("Null Values Check", nulls_df)
    else:
        messagebox.showerror("Error", "No dataset loaded")

def display_info():
    if df is not None:
        # Construct a DataFrame with the info details
        info_data = {
            "Column": df.columns,
            "Non-Null Count": df.notnull().sum(),
            "Data Type": df.dtypes
        }
        info_df = pd.DataFrame(info_data)
        create_table_window("Dataset Info", info_df)  # Use the table window for structured display
    else:
        messagebox.showerror("Error", "No dataset loaded")

def display_describe():
    if df is not None:
        # Get descriptive statistics as a DataFrame
        stats_df = df.describe().reset_index()
        
        # Use the table window to display descriptive statistics
        create_table_window("Descriptive Statistics", stats_df)
    else:
        messagebox.showerror("Error", "No dataset loaded")

def plot_histogram():
    if df is not None:
        fig = plt.figure(figsize=(15, 15))
        ax = fig.gca()
        df.hist(ax=ax)
        plt.show()
    else:
        messagebox.showerror("Error", "No dataset loaded")

def check_balance():
    if df is not None:
        sns.countplot(x='target', data=df)
        plt.xlabel('Target')
        plt.ylabel('Count')
        plt.title('Target Balance')
        plt.show()
    else:
        messagebox.showerror("Error", "No dataset loaded")

def plot_heatmap():
    if df is not None:
        corr_matrix = df.corr()
        plt.figure(figsize=(15, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix Heatmap")
        plt.show()
    else:
        messagebox.showerror("Error", "No dataset loaded")

# Modeling and Prediction
def perform_knn():
    if df is not None:
        knn_scores = []
        for i in range(1, 21):
            knn_classifier = KNeighborsClassifier(n_neighbors=i)
            scores = cross_val_score(knn_classifier, X_scaled, y, cv=10)
            knn_scores.append(scores.mean())
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 21), knn_scores, marker='*', color='red')
        plt.title("KNN Accuracy Scores for Different K")
        plt.xlabel("Number of Neighbors (K)")
        plt.ylabel("Accuracy")
        plt.show()
    else:
        messagebox.showerror("Error", "No dataset loaded")

def perform_decision_tree():
    if df is not None:
        scores = []
        for i in range(1, 11):
            dt = DecisionTreeClassifier(max_depth=i)
            scores.append(cross_val_score(dt, X_scaled, y, cv=10).mean())
        plt.plot(range(1, 11), scores, marker='*', color='green')
        plt.title("Decision Tree Accuracy Scores for Different Depths")
        plt.xlabel("Max Depth")
        plt.ylabel("Accuracy")
        plt.show()
    else:
        messagebox.showerror("Error", "No dataset loaded")

def perform_random_forest():
    if df is not None:
        scores = []
        for i in range(10, 101, 10):
            rf = RandomForestClassifier(n_estimators=i)
            scores.append(cross_val_score(rf, X_scaled, y, cv=5).mean())
        plt.plot(range(10, 101, 10), scores, marker='*', color='blue')
        plt.title("Random Forest Accuracy Scores for Different Trees")
        plt.xlabel("Number of Estimators")
        plt.ylabel("Accuracy")
        plt.show()
    else:
        messagebox.showerror("Error", "No dataset loaded")

def predict_probabilities():
    if df is not None:
        classifier_choice = simpledialog.askstring("Classifier", "Choose a classifier (knn/decision_tree/random_forest):")
        
        if classifier_choice is None:
            return
        
        classifier_choice = classifier_choice.strip().lower()
        
        if classifier_choice not in ['knn', 'decision_tree', 'random_forest']:
            messagebox.showerror("Error", "Invalid classifier choice. Please choose 'knn', 'decision_tree', or 'random_forest'.")
            return

        if classifier_choice == 'knn':
            classifier = KNeighborsClassifier(n_neighbors=5)
        elif classifier_choice == 'decision_tree':
            classifier = DecisionTreeClassifier(max_depth=5)
        else:
            classifier = RandomForestClassifier(n_estimators=50)

        classifier.fit(X_scaled, y)

        user_input = simpledialog.askstring("Input", "Enter feature values separated by commas:")
        if user_input:
            try:
                input_values = [float(x.strip()) for x in user_input.split(',')]
                
                if len(input_values) != X.shape[1]:
                    raise ValueError(f"Expected {X.shape[1]} values, but got {len(input_values)}.")
                
                input_values = np.array(input_values).reshape(1, -1)
                input_values_scaled = scaler.transform(input_values)

                probabilities = classifier.predict_proba(input_values_scaled)[0]
                
                result = (
                    f"Probability of No Heart Disease: {probabilities[0]:.2f}\n"
                    f"Probability of Heart Disease: {probabilities[1]:.2f}"
                )
                messagebox.showinfo("Prediction Result", result)
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid input. {str(e)}")
    else:
        messagebox.showerror("Error", "No dataset loaded")

# GUI Menu
def main_menu():
    root = tk.Tk()
    root.title("Heart Analysis & Model Training Tool")
    root.geometry("600x400")
    root.configure(bg='black')

    menu_bar = tk.Menu(root)

    # Load Dataset Option
    menu_bar.add_command(label="Load Report(csv)", command=load_dataset)

    # Dataset Operations
    dataset_menu = tk.Menu(menu_bar, tearoff=0)
    dataset_menu.add_command(label="Display Shape", command=display_shape)
    dataset_menu.add_command(label="Display Column Headers", command=display_column_headers)
    dataset_menu.add_command(label="Display Data Types", command=display_data_types)
    dataset_menu.add_command(label="Display Head", command=display_head)
    dataset_menu.add_command(label="Display Tail", command=display_tail)
    dataset_menu.add_command(label="Check Null Values", command=check_null_values)
    dataset_menu.add_command(label="Display Info", command=display_info)
    dataset_menu.add_command(label="Display Descriptive Statistics", command=display_describe)
    menu_bar.add_cascade(label="Report Summary", menu=dataset_menu)

    # Visualization Operations
    visualization_menu = tk.Menu(menu_bar, tearoff=0)
    visualization_menu.add_command(label="Plot Histogram", command=plot_histogram)
    visualization_menu.add_command(label="Check Balance", command=check_balance)
    visualization_menu.add_command(label="Plot Heatmap", command=plot_heatmap)
    menu_bar.add_cascade(label="Visualization", menu=visualization_menu)

    # Modeling Operations
    model_menu = tk.Menu(menu_bar, tearoff=0)
    model_menu.add_command(label="Perform KNN", command=perform_knn)
    model_menu.add_command(label="Perform Decision Tree", command=perform_decision_tree)
    model_menu.add_command(label="Perform Random Forest", command=perform_random_forest)
    model_menu.add_command(label="Predict Probabilities", command=predict_probabilities)
    menu_bar.add_cascade(label="Modeling", menu=model_menu)

    # Storage Operations
    menu_bar.add_command(label="Patient's Data", command=storage_window)

    # Exit Option
    menu_bar.add_command(label="Exit", command=root.quit)

    root.config(menu=menu_bar)

     # Load the heart image
    heart_image_path = "C:\\Users\\SHAHID\\Desktop\\heart  EDA\\heartimagepng.png"  # Replace with the actual path to your heart image
    heart_image = tk.PhotoImage(file=heart_image_path)

    # Create a label to display the heart image
    heart_label = tk.Label(root, image=heart_image, bg='black')
    heart_label.pack(pady=5)

    # Create a separate button for predicting probabilities
    predict_button = tk.Button(root, text="Predict Probabilities", command=predict_probabilities, bg='black', fg='white', font=('Helvetica', 12, 'bold'))
    predict_button.pack(pady=20)


    root.mainloop()

if __name__ == "__main__":
    main_menu()

'''from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
'''