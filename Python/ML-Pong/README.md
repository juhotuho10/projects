###  ML-Pong
---
- Created in December 2023
- It's Pong, but with a twist: the ball behaves like a hyperactive bouncy ball, affected by exaggerated physics
- The ball is unpredictable, and as its speed increases over time, the game becomes quite intense

You can play against the ML model, and the game keeps score of points won by each side.

Instructions:
Run Pong_loading.py to play.

Controls:

- w = up
- s = down

ML model vision signals:

- Difference between the ball’s y-coordinate and the center of the paddle, normalized
- Paddle y-coordinate, normalized
- Ball x and y coordinates, normalized
- Ball x and y velocities, normalized

Rewards:

- Hitting the ball with the paddle: (+)
- Scoring a point: (+)
- Opponent scoring a point: (-)