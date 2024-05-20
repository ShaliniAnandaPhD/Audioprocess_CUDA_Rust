// Rust Tutorial: Building a Command-Line Todo Application

// Importing necessary modules
use std::fs::{OpenOptions, File};
use std::io::{self, Write, Read, BufRead, BufReader};
use std::path::Path;

// Main function
fn main() {
    // Displaying the main menu
    loop {
        println!("Todo App");
        println!("1. Add a task");
        println!("2. View tasks");
        println!("3. Remove a task");
        println!("4. Exit");

        let choice = get_user_input("Enter your choice: ");

        match choice.trim() {
            "1" => add_task(),
            "2" => view_tasks(),
            "3" => remove_task(),
            "4" => break,
            _ => println!("Invalid choice, please try again."),
        }
    }
}

// Function to get user input
fn get_user_input(prompt: &str) -> String {
    let mut input = String::new();
    println!("{}", prompt);
    io::stdin().read_line(&mut input).expect("Failed to read line");
    input
}

// Function to add a task
fn add_task() {
    let task = get_user_input("Enter the task: ");
    let mut file = OpenOptions::new().append(true).create(true).open("tasks.txt").expect("Could not open file");
    writeln!(file, "{}", task.trim()).expect("Could not write to file");
    println!("Task added!");
}

// Function to view tasks
fn view_tasks() {
    let file = File::open("tasks.txt").expect("Could not open file");
    let reader = BufReader::new(file);

    for (index, line) in reader.lines().enumerate() {
        let line = line.expect("Could not read line");
        println!("{}. {}", index + 1, line);
    }
}

// Function to remove a task
fn remove_task() {
    let task_number = get_user_input("Enter the task number to remove: ");
    let task_number: usize = task_number.trim().parse().expect("Please enter a valid number");

    let file = File::open("tasks.txt").expect("Could not open file");
    let reader = BufReader::new(file);
    let lines: Vec<String> = reader.lines().map(|line| line.expect("Could not read line")).collect();

    if task_number == 0 || task_number > lines.len() {
        println!("Invalid task number");
        return;
    }

    let new_lines: Vec<String> = lines.into_iter().enumerate().filter(|&(i, _)| i != task_number - 1).map(|(_, line)| line).collect();

    let mut file = OpenOptions::new().write(true).truncate(true).open("tasks.txt").expect("Could not open file");
    for line in new_lines {
        writeln!(file, "{}", line).expect("Could not write to file");
    }

    println!("Task removed!");
}

// Adding a few more detailed functions and comments

// Function to display a greeting
fn display_greeting() {
    println!("Welcome to the Todo App!");
}

// Function to show application info
fn show_info() {
    println!("This is a simple command-line Todo application written in Rust.");
}

// Function to save tasks to file
fn save_tasks(tasks: Vec<String>) {
    let mut file = OpenOptions::new().write(true).truncate(true).open("tasks.txt").expect("Could not open file");
    for task in tasks {
        writeln!(file, "{}", task).expect("Could not write to file");
    }
}

// Function to load tasks from file
fn load_tasks() -> Vec<String> {
    if !Path::new("tasks.txt").exists() {
        return Vec::new();
    }

    let file = File::open("tasks.txt").expect("Could not open file");
    let reader = BufReader::new(file);
    reader.lines().map(|line| line.expect("Could not read line")).collect()
}

// Function to edit a task
fn edit_task() {
    let task_number = get_user_input("Enter the task number to edit: ");
    let task_number: usize = task_number.trim().parse().expect("Please enter a valid number");

    let mut tasks = load_tasks();

    if task_number == 0 || task_number > tasks.len() {
        println!("Invalid task number");
        return;
    }

    let new_task = get_user_input("Enter the new task: ");
    tasks[task_number - 1] = new_task.trim().to_string();
    save_tasks(tasks);

    println!("Task edited!");
}

// Adding a few more detailed functions and comments

// Function to mark a task as completed
fn mark_task_completed() {
    let task_number = get_user_input("Enter the task number to mark as completed: ");
    let task_number: usize = task_number.trim().parse().expect("Please enter a valid number");

    let mut tasks = load_tasks();

    if task_number == 0 || task_number > tasks.len() {
        println!("Invalid task number");
        return;
    }

    tasks[task_number - 1] = format!("{} (completed)", tasks[task_number - 1]);
    save_tasks(tasks);

    println!("Task marked as completed!");
}

// Function to clear all tasks
fn clear_all_tasks() {
    save_tasks(Vec::new());
    println!("All tasks cleared!");
}

// Updating the main menu to include new functionalities
fn main() {
    display_greeting();

    loop {
        println!("Todo App");
        println!("1. Add a task");
        println!("2. View tasks");
        println!("3. Remove a task");
        println!("4. Edit a task");
        println!("5. Mark task as completed");
        println!("6. Clear all tasks");
        println!("7. Info");
        println!("8. Exit");

        let choice = get_user_input("Enter your choice: ");

        match choice.trim() {
            "1" => add_task(),
            "2" => view_tasks(),
            "3" => remove_task(),
            "4" => edit_task(),
            "5" => mark_task_completed(),
            "6" => clear_all_tasks(),
            "7" => show_info(),
            "8" => break,
            _ => println!("Invalid choice, please try again."),
        }
    }
}
