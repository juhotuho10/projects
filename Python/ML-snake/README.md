#  ML-Snake
A modified Snake game driven by an ML algorithm with a custom reward function and vision signals I made in August 2022

The environment evolved through iterative changes, with each new class inheriting from the previous one, resulting in the latest iteration (SnekEnv12) being the most refined

<video src="media/Demo_recording.mp4" width="600" controls></video>

### Running the model
You can run the pre-trained model using snek_game_loading.py to see it in action.

Everything else is already set up; no changes needed.

You can set RENDER = False to quickly run 100 games and print the final length to the console after each death.

The best length achieved was 178.

### Snake vision signals (simplified):

- Position of the apple relative to the head: X and Y [-1, 0, 1] (doesn't give distance, only direction)
- Position of the snake’s middle body part relative to the head: X and Y [-1, 0, 1]
- Position of the snake’s tail relative to the head: X and Y [-1, 0, 1]
- Proximity of "danger tiles" (body parts or edges) in each direction


### Rewards (simplified):

- Moving closer to the apple: (+)
- Moving farther from the apple: (-) (heavier penalty to prevent the snake from circling aimlessly)
- Being far from danger tiles: (+) or close to danger tiles: (-) (encourages safety over taking the shortest route to the apple)
- Reaching the apple: (+)
- Dying: (-)

The vision signals and rewards are relatively simple but required a lot of thought, theory, and trial-and-error to simplify effectively.

Note: The original Snake game was not made by me, but most of it has been replaced, except for rendering. (You can find the original game in original_snake_game.py if you're interested in comparing or trying to outperform the ML algorithm.)

