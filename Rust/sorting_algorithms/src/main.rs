use rand::Rng;
use std::collections::VecDeque;
use std::io::stdin;
use std::time::Instant;

fn main() {
    /*
    Algorithm project i decided to make to get familiar with using Rust

    I have programmed the algorithms from just a description of how they work without looking at actual code
    so they so they arent as efficient as they could be but I learned a lot

    notably my implementation of shell_sort is a lot slower than it should be in the real world

     */
    let mut my_vec: Vec<i32>;
    let mut target_vec: Vec<i32>;
    let mut sorted_vec: Vec<i32>;
    let mut start: Instant;
    let mut duration: f32;
    let mut times: Vec<f32>;

    let fn_names: Vec<&str> = vec![
        "default_rust_sort",
        "comb_sort",
        "quick_sort",
        "merge_sort_deque",
        "merge_sort",
        "bucket_sort",
        "insertion_sort",
        "shell_sort",
        "selection_sort",
        "optimized_bubble_sort",
        "bubble_sort",
    ];

    let functions: Vec<fn(Vec<i32>) -> Vec<i32>> = vec![
        default_rust_sort,
        comb_sort,
        quick_sort,
        merge_sort_deque,
        merge_sort,
        bucket_sort,
        insertion_sort,
        shell_sort,
        selection_sort,
        optimized_bubble_sort,
        bubble_sort,
    ];

    // sizes for vectors that we sort using the functions
    let range_vec = [
        8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
    ];

    println!("sorting vectors sized: {:?}\n", range_vec);

    // enumerating through the functions to conveniently use them without a long main function
    for (fn_id, func) in functions.iter().enumerate() {
        println!("function: {:?} times in ms:", fn_names[fn_id]);
        times = Vec::new();

        for range in range_vec {
            my_vec = make_vec(range);

            target_vec = my_vec.clone();

            // known sorted vector
            target_vec.sort();

            start = Instant::now();

            sorted_vec = func(my_vec);

            duration = start.elapsed().as_nanos() as f32;

            // making sure our vector is the same as a known sorted vector
            assert_eq!(sorted_vec, target_vec);

            // convering the duration to milliseconds while preserving decimals
            duration /= 1_000_000_f32;

            times.push(duration);
        }
        println!("{:?}\n", times);
    }

    println!("done!");

    // user input just to stop the EXE windows from closing right after the program is finished
    let mut input = String::with_capacity(100);
    stdin().read_line(&mut input).unwrap();
}

fn make_vec(x: i32) -> Vec<i32> {
    // generates a random vector of size x with numbers ranging from -x to x

    let vec_size = x as usize;

    let mut num_vec: Vec<i32> = Vec::with_capacity(vec_size);

    let mut rng = rand::thread_rng();

    for _ in 0..num_vec.capacity() {
        num_vec.push(rng.gen_range(-x..=x));
    }

    num_vec
}

fn optimized_bubble_sort(mut num_vec: Vec<i32>) -> Vec<i32> {
    for i in 0..num_vec.len() - 1 {
        for j in 0..num_vec.len() - 1 - i {
            if num_vec[j] > num_vec[j + 1] {
                num_vec.swap(j, j + 1);
            }
        }
    }

    num_vec
}

fn bubble_sort(mut num_vec: Vec<i32>) -> Vec<i32> {
    let mut done: bool = false;

    while !done {
        done = true;
        for i in 0..num_vec.len() - 1 {
            if num_vec[i] > num_vec[i + 1] {
                num_vec.swap(i, i + 1);
                done = false;
            }
        }
    }
    num_vec
}

fn selection_sort(mut num_vec: Vec<i32>) -> Vec<i32> {
    let mut min_index: usize;
    let mut found_index: usize;
    let mut found_min_num: i32;
    let mut curr_num: i32;

    for i in 0..num_vec.len() - 1 {
        min_index = i;
        found_min_num = num_vec[i];
        found_index = i;
        for j in (i + 1)..num_vec.len() {
            curr_num = num_vec[j];
            if curr_num < found_min_num {
                found_min_num = curr_num;
                found_index = j;
            }
        }

        num_vec[found_index] = num_vec[min_index];
        num_vec[min_index] = found_min_num;
    }
    num_vec
}

fn insertion_sort(mut num_vec: Vec<i32>) -> Vec<i32> {
    let mut compare_index: usize;
    let mut current_num: i32;

    for i in 1..num_vec.len() {
        compare_index = i - 1;
        current_num = num_vec.remove(i);

        loop {
            if current_num < num_vec[compare_index] {
                if compare_index > 0 {
                    compare_index -= 1;
                    continue;
                } else {
                    num_vec.insert(compare_index, current_num);
                    break;
                }
            } else {
                num_vec.insert(compare_index + 1, current_num);
                break;
            }
        }
    }

    num_vec
}

fn merge_sort_deque(num_vec: Vec<i32>) -> Vec<i32> {
    let start: usize = 0;
    let end: usize = num_vec.len();
    let middle: usize = (start + end) / 2;
    let mut left_side: Vec<i32> = num_vec[start..middle].to_vec();
    let mut right_side: Vec<i32> = num_vec[middle..end].to_vec();

    if left_side.len() > 1 {
        left_side = merge_sort_deque(left_side);
    }

    if right_side.len() > 1 {
        right_side = merge_sort_deque(right_side);
    }

    let mut left_side: VecDeque<i32> = VecDeque::from(left_side);
    let mut right_side: VecDeque<i32> = VecDeque::from(right_side);

    let mut final_vec: VecDeque<i32> = VecDeque::new();

    let mut first_left = left_side.pop_front();
    let mut first_right = right_side.pop_front();

    loop {
        match (first_left, first_right) {
            (Some(left), Some(right)) => {
                if left < right {
                    final_vec.push_back(left);
                    first_left = left_side.pop_front();
                } else {
                    final_vec.push_back(right);
                    first_right = right_side.pop_front();
                }
            }
            (None, Some(right)) => {
                final_vec.push_back(right);
                final_vec.extend(right_side);
                break;
            }
            (Some(left), None) => {
                final_vec.push_back(left);
                final_vec.extend(left_side);
                break;
            }
            (None, None) => panic!("Both fields empty, shouldn't happen ever"),
        }
    }

    final_vec.into_iter().collect()
}

fn merge_sort(num_vec: Vec<i32>) -> Vec<i32> {
    let start: usize = 0;
    let end: usize = num_vec.len();
    let middle: usize = (start + end) / 2;
    let mut left_side: Vec<i32> = num_vec[start..middle].to_vec();
    let mut right_side: Vec<i32> = num_vec[middle..end].to_vec();

    // division
    if left_side.len() > 1 {
        left_side = merge_sort(left_side);
    }

    if right_side.len() > 1 {
        right_side = merge_sort(right_side);
    }

    let mut final_vec: Vec<i32> = Vec::new();

    let mut left_last: usize = left_side.len() - 1;
    let mut right_last: usize = right_side.len() - 1;

    let mut push_num: i32;

    // merging step
    loop {
        if left_side[0] < right_side[0] {
            push_num = left_side.remove(0);
            final_vec.push(push_num);

            if left_last == 0 {
                final_vec.extend(right_side);
                break;
            } else {
                left_last -= 1;
            }
        } else {
            push_num = right_side.remove(0);
            final_vec.push(push_num);

            if right_last == 0 {
                final_vec.extend(left_side);
                break;
            } else {
                right_last -= 1;
            }
        }
    }
    final_vec
}

fn bucket_sort(mut num_vec: Vec<i32>) -> Vec<i32> {
    let vec_size: usize = num_vec.len();

    let min_opt = num_vec.iter().min();

    let max_opt = num_vec.iter().max();

    let (min_int, max_int) = match (min_opt, max_opt) {
        (Some(min), Some(max)) => (*min, *max),
        (_, _) => panic!("Vector empty"),
    };

    let range: f32 = (max_int - min_int) as f32;

    let mut relative_position: f32;

    let mut approx_index: usize;

    let mut segment_vector: Vec<Vec<i32>> = Vec::new();

    for _ in 0..=vec_size / 10 {
        segment_vector.push(Vec::new());
    }

    let empty_vector_max_index: f32 = ((vec_size / 10) as f32).round();

    for num in num_vec {
        relative_position = (num - min_int) as f32 / range;
        approx_index = (relative_position * empty_vector_max_index) as usize;

        segment_vector[approx_index].push(num);
    }

    num_vec = segment_vector.into_iter().flatten().collect();

    num_vec = insertion_sort(num_vec);

    num_vec
}

fn default_rust_sort(mut num_vec: Vec<i32>) -> Vec<i32> {
    num_vec.sort();

    num_vec
}

fn quick_sort(mut num_vec: Vec<i32>) -> Vec<i32> {
    let vec_len = num_vec.len();

    match vec_len {
        2 => {
            if num_vec[0] > num_vec[1] {
                num_vec.swap(0, 1);
            }
        }

        3.. => {
            let pivot = num_vec[0];
            let mut a_idx: usize = 1;
            let mut b_idx: usize = num_vec.len() - 1;

            while a_idx != b_idx {
                while (num_vec[a_idx] <= pivot) && (a_idx != b_idx) {
                    a_idx += 1;
                }

                while (num_vec[b_idx] > pivot) && (a_idx != b_idx) {
                    b_idx -= 1;
                }
                num_vec.swap(a_idx, b_idx);
            }

            let first_half;
            let second_half;

            if num_vec[a_idx] < pivot {
                num_vec.swap(0, a_idx);
                first_half = num_vec[0..a_idx].to_vec();
                second_half = num_vec[a_idx + 1..].to_vec();
            } else {
                num_vec.swap(0, a_idx - 1);
                first_half = num_vec[0..a_idx - 1].to_vec();
                second_half = num_vec[a_idx..].to_vec();
            }

            num_vec = quick_sort(first_half);

            num_vec.push(pivot);

            num_vec.extend(quick_sort(second_half));
        }
        _ => {}
    }

    num_vec
}

fn comb_sort(mut num_vec: Vec<i32>) -> Vec<i32> {
    let shrinking_factor: f32 = 0.77;
    let mut gap = (num_vec.len() as f32 * shrinking_factor).round() as usize;
    let vec_len = num_vec.len();

    let mut done: bool = false;

    while !done {
        if gap == 1 {
            done = true;
        }
        let mut i: usize = 0;
        while i + gap < vec_len {
            let gap_i = i + gap;
            if num_vec[i] > num_vec[gap_i] {
                num_vec.swap(i, gap_i);
                done = false;
            }
            i += 1;
        }

        gap = (gap as f32 * shrinking_factor).floor() as usize;
        gap = gap.max(1);
    }

    num_vec
}

fn shell_sort(mut num_vec: Vec<i32>) -> Vec<i32> {
    // no idea why my shell sort is so slow, it's supposed to be faster than insertion short, but this is a bit slower?

    let mut compare_index: usize;
    let mut current_num: i32;

    let shrinking_factor: f32 = 2.3;
    let vec_len = num_vec.len();
    let mut gap: usize = (vec_len as f32 / shrinking_factor).round() as usize;
    let mut remainder: usize;

    loop {
        remainder = vec_len % gap;

        for i in 0..=remainder {
            for j in (gap + i..vec_len).step_by(gap) {
                compare_index = j - gap;
                current_num = num_vec.remove(j);

                loop {
                    if current_num < num_vec[compare_index] {
                        if compare_index >= gap {
                            compare_index -= gap;
                            continue;
                        } else {
                            num_vec.insert(compare_index, current_num);
                            break;
                        }
                    } else {
                        num_vec.insert(compare_index + gap, current_num);
                        break;
                    }
                }
            }
        }

        if gap == 1 {
            return num_vec;
        }

        gap = (gap as f32 / shrinking_factor).floor() as usize;
        gap = gap.max(1);
    }
}
