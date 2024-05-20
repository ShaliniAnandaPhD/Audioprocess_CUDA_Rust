// Rust Tutorial

// Entry point for Rust programs
fn main() {
    // 1. Variable declaration
    // Variables are immutable by default
    let x = 5;
    println!("The value of x is: {}", x);

    // To make a variable mutable, use `mut`
    let mut y = 10;
    println!("The value of y is: {}", y);
    y = 20;
    println!("The new value of y is: {}", y);

    // 2. Constants
    // Constants are immutable and must have a type
    const MAX_POINTS: u32 = 100_000;
    println!("The maximum points are: {}", MAX_POINTS);

    // 3. Data Types
    // Scalar types: integers, floating-point numbers, Booleans, and characters
    let a: i32 = -42;
    let b: f64 = 3.14;
    let c: bool = true;
    let d: char = 'z';
    println!("a: {}, b: {}, c: {}, d: {}", a, b, c, d);

    // 4. Compound Types
    // Tuple: a fixed-size collection of values of different types
    let tup: (i32, f64, u8) = (500, 6.4, 1);
    let (x, y, z) = tup; // Destructuring a tuple
    println!("Tuple values: x = {}, y = {}, z = {}", x, y, z);

    // Array: a fixed-size collection of values of the same type
    let arr = [1, 2, 3, 4, 5];
    let first = arr[0];
    println!("The first element of the array is: {}", first);

    // 5. Functions
    // Defining and calling a function
    fn add(a: i32, b: i32) -> i32 {
        a + b
    }
    let sum = add(5, 10);
    println!("The sum is: {}", sum);

    // 6. Control Flow
    // If-else statements
    let number = 6;
    if number % 4 == 0 {
        println!("The number is divisible by 4");
    } else if number % 3 == 0 {
        println!("The number is divisible by 3");
    } else if number % 2 == 0 {
        println!("The number is divisible by 2");
    } else {
        println!("The number is not divisible by 4, 3, or 2");
    }

    // Loops
    // Loop: an infinite loop
    let mut counter = 0;
    let result = loop {
        counter += 1;
        if counter == 10 {
            break counter * 2; // Exits the loop
        }
    };
    println!("The result is: {}", result);

    // While loop
    let mut number = 3;
    while number != 0 {
        println!("{}!", number);
        number -= 1;
    }
    println!("LIFTOFF!!!");

    // For loop
    let a = [10, 20, 30, 40, 50];
    for element in a.iter() {
        println!("The value is: {}", element);
    }

    // 7. Ownership
    // Ownership rules: each value in Rust has a variable that's called its owner
    // There can only be one owner at a time
    // When the owner goes out of scope, the value will be dropped

    // String type
    let s1 = String::from("hello");
    let s2 = s1; // s1 is moved to s2, s1 is no longer valid
    println!("s2: {}", s2);

    // Cloning a string
    let s1 = String::from("hello");
    let s2 = s1.clone(); // Deep copy of s1
    println!("s1: {}, s2: {}", s1, s2);

    // 8. References and Borrowing
    // References allow you to refer to some value without taking ownership
    let s1 = String::from("hello");
    let len = calculate_length(&s1); // Pass a reference to s1
    println!("The length of '{}' is {}", s1, len);

    fn calculate_length(s: &String) -> usize {
        s.len()
    }

    // Mutable references
    let mut s = String::from("hello");
    change(&mut s); // Pass a mutable reference to s
    println!("The changed string is: {}", s);

    fn change(some_string: &mut String) {
        some_string.push_str(", world");
    }

    // 9. Slices
    // Slices let you reference a contiguous sequence of elements in a collection rather than the whole collection
    let s = String::from("hello world");
    let hello = &s[0..5]; // Slicing the string
    let world = &s[6..11];
    println!("{} {}", hello, world);

    // 10. Structs
    // Structs are used to create custom data types
    struct User {
        username: String,
        email: String,
        sign_in_count: u64,
        active: bool,
    }

    let user1 = User {
        email: String::from("someone@example.com"),
        username: String::from("someusername123"),
        active: true,
        sign_in_count: 1,
    };
    println!("User: {}", user1.username);

    // 11. Enums
    // Enums allow you to define a type by enumerating its possible values
    enum Message {
        Quit,
        Move { x: i32, y: i32 },
        Write(String),
        ChangeColor(i32, i32, i32),
    }

    fn process_message(msg: Message) {
        match msg {
            Message::Quit => println!("Quit message"),
            Message::Move { x, y } => println!("Move to x: {}, y: {}", x, y),
            Message::Write(text) => println!("Text message: {}", text),
            Message::ChangeColor(r, g, b) => println!("Change color to red: {}, green: {}, blue: {}", r, g, b),
        }
    }

    let msg = Message::Write(String::from("Hello, Rust!"));
    process_message(msg);

    // 12. Modules
    // Modules let you organize your code into groups
    mod front_of_house {
        pub mod hosting {
            pub fn add_to_waitlist() {
                println!("Added to waitlist");
            }
        }
    }

    use crate::front_of_house::hosting;
    hosting::add_to_waitlist();

    // 13. Result and Option
    // The Result type is used for error handling
    use std::fs::File;

    let f = File::open("hello.txt");
    let f = match f {
        Ok(file) => file,
        Err(error) => {
            println!("Error opening file: {:?}", error);
            return;
        },
    };

    // The Option type is used for values that might be null
    let some_number = Some(5);
    let absent_number: Option<i32> = None;
    println!("Some number: {:?}", some_number);
    println!("Absent number: {:?}", absent_number);
}
