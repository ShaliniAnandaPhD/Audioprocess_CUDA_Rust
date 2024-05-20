// Rust Programming Questions and Answers

// Question 1
// Q: How do you declare a variable in Rust?
let x = 5;

// Question 2
// Q: How do you make a variable mutable in Rust?
let mut y = 5;
y = 6;

// Question 3
// Q: How do you define a function in Rust?
fn say_hello() {
    println!("Hello, world!");
}

// Question 4
// Q: How do you return a value from a function in Rust?
fn add(a: i32, b: i32) -> i32 {
    a + b
}

// Question 5
// Q: How do you create a constant in Rust?
const MAX_POINTS: u32 = 100_000;

// Question 6
// Q: How do you create a tuple in Rust?
let tup: (i32, f64, u8) = (500, 6.4, 1);

// Question 7
// Q: How do you access elements of a tuple in Rust?
let (x, y, z) = tup;
println!("The value of y is: {}", y);

// Question 8
// Q: How do you create an array in Rust?
let a = [1, 2, 3, 4, 5];

// Question 9
// Q: How do you access elements of an array in Rust?
let first = a[0];
let second = a[1];

// Question 10
// Q: How do you use an if statement in Rust?
if x < 5 {
    println!("x is less than 5");
} else {
    println!("x is not less than 5");
}

// Question 11
// Q: How do you use a for loop in Rust?
let a = [10, 20, 30, 40, 50];
for element in a.iter() {
    println!("the value is: {}", element);
}

// Question 12
// Q: How do you use a while loop in Rust?
let mut number = 3;
while number != 0 {
    println!("{}!", number);
    number -= 1;
}
println!("LIFTOFF!!!");

// Question 13
// Q: How do you use a loop in Rust?
let mut counter = 0;
let result = loop {
    counter += 1;
    if counter == 10 {
        break counter * 2;
    }
};
println!("The result is {}", result);

// Question 14
// Q: How do you define an enum in Rust?
enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(i32, i32, i32),
}

// Question 15
// Q: How do you match on an enum in Rust?
let msg = Message::ChangeColor(0, 160, 255);
match msg {
    Message::Quit => {
        println!("The Quit variant has no data to destructure.");
    }
    Message::Move { x, y } => {
        println!("Move to x: {}, y: {}", x, y);
    }
    Message::Write(text) => {
        println!("Text message: {}", text);
    }
    Message::ChangeColor(r, g, b) => {
        println!("Change the color to red: {}, green: {}, and blue: {}", r, g, b);
    }
}

// Question 16
// Q: How do you define a struct in Rust?
struct User {
    username: String,
    email: String,
    sign_in_count: u64,
    active: bool,
}

// Question 17
// Q: How do you create an instance of a struct in Rust?
let user1 = User {
    email: String::from("someone@example.com"),
    username: String::from("someusername123"),
    active: true,
    sign_in_count: 1,
};

// Question 18
// Q: How do you implement a method for a struct in Rust?
impl User {
    fn email(&self) -> &String {
        &self.email
    }
}
println!("User email: {}", user1.email());

// Question 19
// Q: How do you define a generic function in Rust?
fn largest<T: PartialOrd>(list: &[T]) -> &T {
    let mut largest = &list[0];
    for item in list {
        if item > largest {
            largest = item;
        }
    }
    largest
}

// Question 20
// Q: How do you define a generic struct in Rust?
struct Point<T> {
    x: T,
    y: T,
}

// Question 21
// Q: How do you define an implementation for a generic struct in Rust?
impl<T> Point<T> {
    fn x(&self) -> &T {
        &self.x
    }
}

// Question 22
// Q: How do you define a trait in Rust?
trait Summary {
    fn summarize(&self) -> String;
}

// Question 23
// Q: How do you implement a trait for a struct in Rust?
impl Summary for User {
    fn summarize(&self) -> String {
        format!("{} ({})", self.username, self.email)
    }
}
println!("Summary: {}", user1.summarize());

// Question 24
// Q: How do you use the `Option` enum in Rust?
let some_number = Some(5);
let some_string = Some("a string");
let absent_number: Option<i32> = None;

// Question 25
// Q: How do you use the `Result` enum in Rust?
use std::fs::File;
let f = File::open("hello.txt");
let f = match f {
    Ok(file) => file,
    Err(error) => {
        panic!("There was a problem opening the file: {:?}", error)
    },
};

// Question 26
// Q: How do you use the `?` operator in Rust?
fn read_username_from_file() -> Result<String, std::io::Error> {
    let mut f = File::open("hello.txt")?;
    let mut s = String::new();
    f.read_to_string(&mut s)?;
    Ok(s)
}

// Question 27
// Q: How do you create a new thread in Rust?
use std::thread;
let handle = thread::spawn(|| {
    for i in 1..10 {
        println!("hi number {} from the spawned thread!", i);
        thread::sleep(std::time::Duration::from_millis(1));
    }
});
handle.join().unwrap();

// Question 28
// Q: How do you use a channel in Rust?
use std::sync::mpsc;
let (tx, rx) = mpsc::channel();
thread::spawn(move || {
    let val = String::from("hi");
    tx.send(val).unwrap();
});
let received = rx.recv().unwrap();
println!("Got: {}", received);

// Question 29
// Q: How do you use a Mutex in Rust?
use std::sync::Mutex;
let m = Mutex::new(5);
{
    let mut num = m.lock().unwrap();
    *num = 6;
}
println!("m = {:?}", m);

// Question 30
// Q: How do you define a vector in Rust?
let v: Vec<i32> = Vec::new();

// Question 31
// Q: How do you add elements to a vector in Rust?
let mut v = Vec::new();
v.push(5);
v.push(6);
v.push(7);

// Question 32
// Q: How do you read elements from a vector in Rust?
let third: &i32 = &v[2];
println!("The third element is {}", third);

// Question 33
// Q: How do you iterate over elements in a vector in Rust?
for i in &v {
    println!("{}", i);
}

// Question 34
// Q: How do you use a `HashMap` in Rust?
use std::collections::HashMap;
let mut scores = HashMap::new();
scores.insert(String::from("Blue"), 10);
scores.insert(String::from("Yellow"), 50);

// Question 35
// Q: How do you get a value from a `HashMap` in Rust?
let team_name = String::from("Blue");
let score = scores.get(&team_name);
println!("Score: {:?}", score);

// Question 36
// Q: How do you iterate over key-value pairs in a `HashMap` in Rust?
for (key, value) in &scores {
    println!("{}: {}", key, value);
}

// Question 37
// Q: How do you update a value in a `HashMap` in Rust?
scores.insert(String::from("Blue"), 25);

// Question 38
// Q: How do you update a value based on the old value in a `HashMap` in Rust?
let team_name = String::from("Blue");
let score = scores.entry(team_name).or_insert(50);
*score += 10;
println!("Updated score: {:?}", scores.get(&String::from("Blue")));

// Question 39
// Q: How do you define an iterator in Rust?
let v = vec![1, 2, 3];
let v_iter = v.iter();

// Question 40
// Q: How do you use the `map` method on an iterator in Rust?
let v: Vec<i32> = vec![1, 2, 3];
let v2: Vec<_> = v.iter().map(|x| x + 1).collect();
println!("Updated vector: {:?}", v2);

// Question 41
// Q: How do you use the `filter` method on an iterator in Rust?
let v: Vec<i32> = vec![1, 2, 3, 4, 5];
let evens: Vec<_> = v.into_iter().filter(|x| x % 2 == 0).collect();
println!("Even numbers: {:?}", evens);

// Question 42
// Q: How do you define a closure in Rust?
let add_one = |x: i32| -> i32 { x + 1 };
println!("Result: {}", add_one(5));

// Question 43
// Q: How do you capture variables in a closure in Rust?
let x = 4;
let equal_to_x = |z| z == x;
let y = 4;
assert!(equal_to_x(y));

// Question 44
// Q: How do you define a struct that holds a closure in Rust?
struct Cacher<T>
where
    T: Fn(u32) -> u32,
{
    calculation: T,
    value: Option<u32>,
}

// Question 45
// Q: How do you define an implementation for a struct that holds a closure in Rust?
impl<T> Cacher<T>
where
    T: Fn(u32) -> u32,
{
    fn new(calculation: T) -> Cacher<T> {
        Cacher {
            calculation,
            value: None,
        }
    }

    fn value(&mut self, arg: u32) -> u32 {
        match self.value {
            Some(v) => v,
            None => {
                let v = (self.calculation)(arg);
                self.value = Some(v);
                v
            }
        }
    }
}

// Question 46
// Q: How do you define a macro in Rust?
macro_rules! say_hello {
    () => {
        println!("Hello!");
    };
}
say_hello!();

// Question 47
// Q: How do you define a macro that takes arguments in Rust?
macro_rules! create_function {
    ($func_name:ident) => {
        fn $func_name() {
            println!("You called {:?}()!", stringify!($func_name));
        }
    };
}
create_function!(foo);
foo();

// Question 48
// Q: How do you read a file in Rust?
use std::fs;
let contents = fs::read_to_string("hello.txt")
    .expect("Something went wrong reading the file");
println!("File contents: {}", contents);

// Question 49
// Q: How do you write to a file in Rust?
use std::io::Write;
let mut file = File::create("output.txt").expect("Could not create file");
file.write_all(b"Hello, world!").expect("Could not write to file");

// Question 50
// Q: How do you define and use an attribute in Rust?
#[derive(Debug)]
struct Person {
    name: String,
    age: u8,
}
let person = Person {
    name: String::from("Alice"),
    age: 30,
};
println!("{:?}", person);
