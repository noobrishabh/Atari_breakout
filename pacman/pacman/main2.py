import pygame
import sys
import random
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import pickle
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(
    filename='game.log',
    level=logging.DEBUG,  # Change to INFO or WARNING in production
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Adaptive Pac-Man with Q-Learning AI")

# Q-Learning Parameters
ALPHA = 0.1       # Learning rate
GAMMA = 0.9       # Discount factor
EPSILON = 1.0     # Exploration rate
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

# Define actions
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']

# Grid size for state representation
GRID_SIZE = 20
NUM_GRID_X = SCREEN_WIDTH // GRID_SIZE
NUM_GRID_Y = SCREEN_HEIGHT // GRID_SIZE

# Define fuzzy variables for difficulty (if still needed)
# Define fuzzy variables
score = ctrl.Antecedent(np.arange(0, 1001, 1), 'score')
lives = ctrl.Antecedent(np.arange(0, 4, 1), 'lives')
ghost_proximity = ctrl.Antecedent(np.arange(0, 300, 1), 'ghost_proximity')

difficulty = ctrl.Consequent(np.arange(0.5, 3.1, 0.1), 'difficulty')

# Define membership functions for score
score['low'] = fuzz.trimf(score.universe, [0, 0, 500])
score['medium'] = fuzz.trimf(score.universe, [300, 500, 700])
score['high'] = fuzz.trimf(score.universe, [600, 1000, 1000])

# Define membership functions for lives
lives['few'] = fuzz.trimf(lives.universe, [0, 0, 1])
lives['some'] = fuzz.trimf(lives.universe, [0, 2, 3])
lives['many'] = fuzz.trimf(lives.universe, [2, 3, 3])

# Define membership functions for ghost proximity
ghost_proximity['close'] = fuzz.trimf(ghost_proximity.universe, [0, 0, 150])
ghost_proximity['medium'] = fuzz.trimf(ghost_proximity.universe, [100, 150, 200])
ghost_proximity['far'] = fuzz.trimf(ghost_proximity.universe, [150, 300, 300])

# Define membership functions for difficulty
difficulty['easy'] = fuzz.trimf(difficulty.universe, [0.5, 0.5, 1.5])
difficulty['medium'] = fuzz.trimf(difficulty.universe, [1.0, 1.5, 2.0])
difficulty['hard'] = fuzz.trimf(difficulty.universe, [1.5, 3.0, 3.0])

# Define fuzzy rules
rule1 = ctrl.Rule(score['high'] & lives['many'] & ghost_proximity['far'], difficulty['hard'])
rule2 = ctrl.Rule(score['high'] & (lives['some'] | ghost_proximity['medium']), difficulty['medium'])
rule3 = ctrl.Rule(score['medium'] & lives['some'] & ghost_proximity['medium'], difficulty['medium'])
rule4 = ctrl.Rule(score['medium'] & (lives['few'] | ghost_proximity['close']), difficulty['easy'])
rule5 = ctrl.Rule(score['low'] | lives['few'] | ghost_proximity['close'], difficulty['easy'])

difficulty_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])

def load_and_scale_image(path, size, placeholder_color):
    """
    Loads an image from the specified path and scales it to the given size.
    If the image cannot be loaded, returns a colored circle as a placeholder.
    """
    try:
        image = pygame.image.load(path).convert_alpha()
        return pygame.transform.scale(image, size)
    except pygame.error:
        logging.warning(f"Unable to load image at {path}. Using placeholder.")
        placeholder = pygame.Surface(size, pygame.SRCALPHA)
        pygame.draw.circle(placeholder, placeholder_color, (size[0]//2, size[1]//2), size[0]//2)
        return placeholder

# Initialize Q-Table
Q_TABLE = defaultdict(lambda: {action: 0.0 for action in ACTIONS})

class PacMan(pygame.sprite.Sprite):
    def __init__(self, pos, color=(0, 255, 255)):
        super().__init__()
        # Attempt to load Pac-Man image; use colored circle if unavailable
        self.image = load_and_scale_image('assets/pacman.png', (30, 30), color)
        self.rect = self.image.get_rect(center=pos)
        self.speed = 4
        self.direction = pygame.math.Vector2(1, 0)  # Start moving to the right
        self.state = self.get_state([])  # Initialize with empty ghosts
        self.previous_state = None
        self.previous_action = None

    def get_state(self, ghosts_group):
        """
        Discretize the environment into a grid and encode the state based on
        Pac-Man's position and nearby ghosts.
        """
        grid_x = self.rect.centerx // GRID_SIZE
        grid_y = self.rect.centery // GRID_SIZE

        # Find the closest ghost
        if ghosts_group:
            closest_ghost = min(ghosts_group, key=lambda ghost: pygame.math.Vector2(ghost.rect.center).distance_to(self.rect.center))
            ghost_dx = (closest_ghost.rect.centerx - self.rect.centerx) // GRID_SIZE
            ghost_dy = (closest_ghost.rect.centery - self.rect.centery) // GRID_SIZE
            # Limit the range to prevent an excessively large state space
            ghost_dx = max(-5, min(5, ghost_dx))
            ghost_dy = max(-5, min(5, ghost_dy))
        else:
            ghost_dx, ghost_dy = 0, 0

        state = (int(grid_x), int(grid_y), int(ghost_dx), int(ghost_dy))
        return state

    def choose_action(self):
        """
        Choose an action based on the current state using an epsilon-greedy policy.
        """
        global EPSILON

        if random.uniform(0, 1) < EPSILON:
            action = random.choice(ACTIONS)
            logging.debug(f"Exploring: Chose random action {action}")
        else:
            state_actions = Q_TABLE[self.state]
            max_q = max(state_actions.values())
            # In case multiple actions have the same max Q-value, randomly choose among them
            actions_with_max_q = [action for action, q in state_actions.items() if q == max_q]
            action = random.choice(actions_with_max_q)
            logging.debug(f"Exploiting: Chose best action {action} with Q-value {max_q}")
        return action

    def update_q_table(self, reward, done):
        """
        Update the Q-table based on the action taken and the reward received.
        """
        if self.previous_state is not None and self.previous_action is not None:
            old_value = Q_TABLE[self.previous_state][self.previous_action]
            next_max = max(Q_TABLE[self.state].values()) if not done else 0
            # Q-Learning formula
            new_value = old_value + ALPHA * (reward + GAMMA * next_max - old_value)
            Q_TABLE[self.previous_state][self.previous_action] = new_value
            logging.debug(f"Updated Q-value for state {self.previous_state} and action {self.previous_action}: {new_value}")

    def update(self, ghosts_group):
        """
        Update Pac-Man's position based on the chosen action and perform Q-Learning updates.
        """
        global EPSILON

        self.previous_state = self.state
        action = self.choose_action()
        self.previous_action = action

        # Perform the action
        if action == 'UP':
            self.direction = pygame.math.Vector2(0, -1)
        elif action == 'DOWN':
            self.direction = pygame.math.Vector2(0, 1)
        elif action == 'LEFT':
            self.direction = pygame.math.Vector2(-1, 0)
        elif action == 'RIGHT':
            self.direction = pygame.math.Vector2(1, 0)
        
        # Move Pac-Man
        if self.direction.length() > 0:
            self.direction = self.direction.normalize()
            self.rect.x += self.direction.x * self.speed
            self.rect.y += self.direction.y * self.speed

        # Keep Pac-Man within screen bounds
        self.rect.clamp_ip(SCREEN.get_rect())

        # Get the new state after moving
        self.state = self.get_state(ghosts_group)

        # Calculate reward
        reward = 1  # Small reward for each step survived

        # Check for collision with ghosts
        collision = False
        for ghost in ghosts_group:
            distance = pygame.math.Vector2(self.rect.center).distance_to(ghost.rect.center)
            if distance < 25:
                collision = True
                break

        done = False
        if collision:
            reward = -100  # Large negative reward for collision
            done = True

        # Update Q-table
        self.update_q_table(reward, done)

        # Decay epsilon
        if EPSILON > MIN_EPSILON:
            EPSILON *= EPSILON_DECAY
            EPSILON = max(MIN_EPSILON, EPSILON)
            logging.debug(f"Epsilon decayed to {EPSILON}")

        return done, reward

class Ghost(pygame.sprite.Sprite):
    def __init__(self, pos, ghost_type='chaser', difficulty=1.0):
        super().__init__()
        # Attempt to load Ghost image; use red circle if unavailable
        self.image = load_and_scale_image('assets/ghost.png', (30, 30), (255, 0, 0))
        self.rect = self.image.get_rect(center=pos)
        self.base_speed = 2
        self.speed = self.base_speed * difficulty
        self.ghost_type = ghost_type
        self.direction = pygame.math.Vector2(0, 0)

    def update(self, pacman_pos, pacman_direction, difficulty=1.0):
        logging.debug(f"Updating Ghost: {self.ghost_type} at {self.rect.center}")
        self.speed = self.base_speed * difficulty
        if self.ghost_type == 'chaser':
            self.chase(pacman_pos)
        elif self.ghost_type == 'ambusher':
            self.ambush(pacman_pos, pacman_direction)
        elif self.ghost_type == 'random':
            self.random_move()

    def chase(self, pacman_pos):
        # Move directly towards Pac-Man
        direction_vector = pygame.math.Vector2(pacman_pos) - pygame.math.Vector2(self.rect.center)
        if direction_vector.length() > 0:
            direction_vector = direction_vector.normalize()
            self.direction = direction_vector
        else:
            self.direction = pygame.math.Vector2(0, 0)
        self.rect.x += self.direction.x * self.speed
        self.rect.y += self.direction.y * self.speed
        logging.debug(f"Ghost Chaser Moving Towards: {self.direction}")

    def ambush(self, pacman_pos, pacman_direction):
        # Predict Pac-Man's movement and position accordingly
        predicted_pos = pygame.math.Vector2(pacman_pos) + pacman_direction * 50
        direction_vector = predicted_pos - pygame.math.Vector2(self.rect.center)
        if direction_vector.length() > 0:
            direction_vector = direction_vector.normalize()
            self.direction = direction_vector
        else:
            self.direction = pygame.math.Vector2(0, 0)
        self.rect.x += self.direction.x * self.speed
        self.rect.y += self.direction.y * self.speed
        logging.debug(f"Ghost Ambusher Moving Towards: {self.direction}")

    def random_move(self):
        # Move in a random direction
        if random.randint(0, 100) < 5:  # 5% chance to change direction each frame
            self.direction = pygame.math.Vector2(random.choice([-1, 0, 1]), random.choice([-1, 0, 1]))
            if self.direction.length() == 0:
                self.direction = pygame.math.Vector2(1, 0)  # Default to right if no movement
            else:
                self.direction = self.direction.normalize()
            logging.debug(f"Ghost Randomly Changing Direction: {self.direction}")
        self.rect.x += self.direction.x * self.speed
        self.rect.y += self.direction.y * self.speed
        logging.debug(f"Ghost Random Move Direction: {self.direction}")

def calculate_difficulty(current_score, current_lives, average_ghost_distance):
    """
    Calculates the current difficulty level based on the AI Pac-Man's score, lives, and ghost proximity using fuzzy logic.
    """
    sim = ctrl.ControlSystemSimulation(difficulty_ctrl)
    sim.input['score'] = current_score
    sim.input['lives'] = current_lives
    sim.input['ghost_proximity'] = average_ghost_distance

    # Perform fuzzy computation
    try:
        sim.compute()
        difficulty_level = sim.output['difficulty']
        logging.debug(f"Fuzzy Logic Output: {sim.output}")
    except Exception as e:
        logging.warning(f"Fuzzy computation failed: {e}. Using default difficulty 1.0.")
        difficulty_level = 1.0  # Default difficulty

    return difficulty_level

def generate_ghost_position(pacman_pos, min_distance=120):
    """
    Generates a random position for a ghost ensuring it is at least min_distance away from Pac-Man.
    """
    attempts = 0
    max_attempts = 100
    while attempts < max_attempts:
        x = random.randint(50, SCREEN_WIDTH - 50)
        y = random.randint(50, SCREEN_HEIGHT - 50)
        distance = pygame.math.Vector2(x, y).distance_to(pacman_pos)
        if distance >= min_distance:
            return (x, y)
        attempts += 1
    # Fallback position if suitable position not found
    logging.warning("Could not find a suitable ghost position after multiple attempts. Using Pac-Man's position.")
    return pacman_pos  # This may cause collision; better to handle appropriately

def main():
    global Q_TABLE, EPSILON
    clock = pygame.time.Clock()

    # Create sprite groups
    all_sprites = pygame.sprite.Group()
    ghosts = pygame.sprite.Group()

    # Initial game state
    current_score = 0
    current_lives = 3
    average_ghost_distance = 150  # Initialize with a default value

    # Create Ghosts with diverse behaviors first
    pacman_start_pos = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    ghost_types = ['chaser', 'ambusher', 'random', 'chaser']  # Assign different types
    for i in range(4):
        ghost_pos = generate_ghost_position(pacman_start_pos, min_distance=120)
        ghost = Ghost(ghost_pos, ghost_type=ghost_types[i % len(ghost_types)])
        ghosts.add(ghost)
        all_sprites.add(ghost)
        logging.debug(f"Ghost {i+1} spawned at {ghost_pos} as {ghost.ghost_type}")

    # Now create AI Pac-Man
    ai_pacman = PacMan(pacman_start_pos, color=(0, 255, 255))  # Cyan color for AI
    all_sprites.add(ai_pacman)

    # Font for displaying score and lives
    font = pygame.font.SysFont(None, 36)

    # Game Over flag
    game_over = False

    # Collision Threshold
    collision_threshold = 25  # Adjust based on sprite sizes

    # Invincibility variables to prevent rapid consecutive collisions
    invincible = False
    invincibility_timer = 0
    INVINCIBILITY_DURATION = 2000  # in milliseconds

    # Load Q-Table if exists
    try:
        with open('q_table.pkl', 'rb') as f:
            Q_TABLE = pickle.load(f)
            logging.info("Loaded existing Q-Table.")
    except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
        logging.warning(f"Unable to load Q-Table ({e}). Starting fresh.")
        Q_TABLE = defaultdict(lambda: {action: 0.0 for action in ACTIONS})

    while True:
        dt = clock.tick(60)  # Delta time in milliseconds

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Save Q-Table before exiting
                try:
                    with open('q_table.pkl', 'wb') as f:
                        pickle.dump(Q_TABLE, f)
                        logging.info("Q-Table saved.")
                except Exception as e:
                    logging.error(f"Failed to save Q-Table: {e}")
                pygame.quit()
                sys.exit()

        if not game_over:
            # Update AI Pac-Man
            done, reward = ai_pacman.update(ghosts)

            # Check if collision occurred
            if done:
                current_lives -= 1
                logging.info(f"Collision detected! Lives remaining: {current_lives}")
                if current_lives <= 0:
                    logging.info("Game Over!")
                    game_over = True
                else:
                    ai_pacman.rect.center = pacman_start_pos  # Reset Pac-Man position
                    logging.info("AI Pac-Man position reset.")
                    # Reposition ghosts safely
                    for i, ghost in enumerate(ghosts):
                        new_pos = generate_ghost_position(pacman_start_pos, min_distance=120)
                        ghost.rect.center = new_pos
                        logging.debug(f"Ghost {i+1} repositioned to {new_pos}")
                    # Activate invincibility
                    invincible = True
                    invincibility_timer = INVINCIBILITY_DURATION
                    logging.debug("AI Pac-Man is now invincible.")
            else:
                current_score += 1  # Increment score for each step survived

            # Calculate average ghost distance
            distances = [pygame.math.Vector2(ghost.rect.center).distance_to(ai_pacman.rect.center) for ghost in ghosts]
            average_ghost_distance = sum(distances) / len(distances) if distances else 150

            # Calculate difficulty
            difficulty_level = calculate_difficulty(current_score, current_lives, average_ghost_distance)
            logging.debug(f"Calculated Difficulty Level: {difficulty_level}")

            # Optionally cap the difficulty to prevent ghosts from becoming too fast
            MAX_DIFFICULTY = 2.5
            if difficulty_level > MAX_DIFFICULTY:
                difficulty_level = MAX_DIFFICULTY

            # Update ghosts with new difficulty
            for ghost in ghosts:
                ghost.update(ai_pacman.rect.center, ai_pacman.direction, difficulty_level)

            # Handle invincibility timer
            if invincible:
                invincibility_timer -= dt
                if invincibility_timer <= 0:
                    invincible = False
                    logging.debug("Invincibility ended.")

            # Render everything
            SCREEN.fill((0, 0, 0))
            all_sprites.draw(SCREEN)

            # Optionally, draw the minimum distance circle
            MIN_DISTANCE = 120
            pygame.draw.circle(SCREEN, (0, 0, 255), ai_pacman.rect.center, MIN_DISTANCE, 1)

            # Display score and lives
            score_text = font.render(f"Score: {current_score}", True, (255, 255, 255))
            lives_text = font.render(f"Lives: {current_lives}", True, (255, 255, 255))
            SCREEN.blit(score_text, (10, 10))
            SCREEN.blit(lives_text, (10, 50))

            pygame.display.flip()

        else:
            # Game Over Screen
            SCREEN.fill((0, 0, 0))
            game_over_text = font.render("Game Over!", True, (255, 0, 0))
            final_score_text = font.render(f"Final Score: {current_score}", True, (255, 255, 255))
            SCREEN.blit(game_over_text, (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, SCREEN_HEIGHT // 2 - 50))
            SCREEN.blit(final_score_text, (SCREEN_WIDTH // 2 - final_score_text.get_width() // 2, SCREEN_HEIGHT // 2))
            pygame.display.flip()

            # Wait for a few seconds before resetting the game
            pygame.time.delay(3000)
            # Reset game
            game_over = False
            current_score = 0
            current_lives = 3
            ai_pacman.rect.center = pacman_start_pos
            logging.info("Game reset.")
            # Reposition ghosts safely
            for i, ghost in enumerate(ghosts):
                new_pos = generate_ghost_position(pacman_start_pos, min_distance=120)
                ghost.rect.center = new_pos
                logging.debug(f"Ghost {i+1} repositioned to {new_pos}")
            EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)  # Decay epsilon after game over

if __name__ == "__main__":
    main()
