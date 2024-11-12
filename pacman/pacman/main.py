import pygame
import sys
import random
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Adaptive Pac-Man with Fuzzy AI")

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
        print(f"Warning: Unable to load image at {path}. Using placeholder.")
        placeholder = pygame.Surface(size, pygame.SRCALPHA)
        pygame.draw.circle(placeholder, placeholder_color, (size[0]//2, size[1]//2), size[0]//2)
        return placeholder

class PacMan(pygame.sprite.Sprite):
    def __init__(self, pos):
        super().__init__()
        # Attempt to load Pac-Man image; use yellow circle if unavailable
        self.image = load_and_scale_image('assets/pacman.png', (30, 30), (255, 255, 0))
        self.rect = self.image.get_rect(center=pos)
        self.speed = 4
        self.direction = pygame.math.Vector2(0, 0)

    def update(self, keys_pressed):
        self.direction = pygame.math.Vector2(0, 0)
        if keys_pressed[pygame.K_LEFT]:
            self.direction.x = -1
        if keys_pressed[pygame.K_RIGHT]:
            self.direction.x = 1
        if keys_pressed[pygame.K_UP]:
            self.direction.y = -1
        if keys_pressed[pygame.K_DOWN]:
            self.direction.y = 1

        if self.direction.length() > 0:
            self.direction = self.direction.normalize()

        self.rect.x += self.direction.x * self.speed
        self.rect.y += self.direction.y * self.speed

        # Keep Pac-Man within screen bounds
        self.rect.clamp_ip(SCREEN.get_rect())

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
        self.rect.x += direction_vector.x * self.speed
        self.rect.y += direction_vector.y * self.speed

    def ambush(self, pacman_pos, pacman_direction):
        # Predict Pac-Man's movement and position accordingly
        # For simplicity, we'll assume Pac-Man is moving in a straight line and predict a future position
        predicted_pos = pygame.math.Vector2(pacman_pos) + pacman_direction * 50
        direction_vector = predicted_pos - pygame.math.Vector2(self.rect.center)
        if direction_vector.length() > 0:
            direction_vector = direction_vector.normalize()
        self.rect.x += direction_vector.x * self.speed
        self.rect.y += direction_vector.y * self.speed

    def random_move(self):
        # Move in a random direction
        if random.randint(0, 100) < 5:  # 5% chance to change direction each frame
            self.direction = pygame.math.Vector2(random.choice([-1, 0, 1]), random.choice([-1, 0, 1]))
            if self.direction.length() == 0:
                self.direction = pygame.math.Vector2(1, 0)  # Default to right if no movement
            else:
                self.direction = self.direction.normalize()
        self.rect.x += self.direction.x * self.speed
        self.rect.y += self.direction.y * self.speed

def calculate_difficulty(current_score, current_lives, average_ghost_distance):
    """
    Calculates the current difficulty level based on the player's score, lives, and ghost proximity using fuzzy logic.
    """
    sim = ctrl.ControlSystemSimulation(difficulty_ctrl)
    sim.input['score'] = current_score
    sim.input['lives'] = current_lives
    sim.input['ghost_proximity'] = average_ghost_distance

    # Perform fuzzy computation
    sim.compute()
    
    # Debugging: Print the output
    print(f"Fuzzy Logic Output: {sim.output}")
    
    # Handle cases where 'difficulty' might not be set
    if 'difficulty' in sim.output:
        return sim.output['difficulty']
    else:
        print("Warning: 'difficulty' not set in fuzzy logic output. Using default value 1.0.")
        return 1.0  # Default difficulty

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
    print("Warning: Could not find a suitable ghost position after multiple attempts.")
    return pacman_pos  # This may cause collision; better to handle appropriately

def main():
    clock = pygame.time.Clock()

    # Create sprite groups
    all_sprites = pygame.sprite.Group()
    ghosts = pygame.sprite.Group()

    # Initial game state
    current_score = 0
    current_lives = 3
    average_ghost_distance = 150  # Initialize with a default value

    # Create Pac-Man
    pacman_start_pos = (SCREEN_WIDTH//2, SCREEN_HEIGHT//2)
    pacman = PacMan(pacman_start_pos)
    all_sprites.add(pacman)

    # Create Ghosts with diverse behaviors
    ghost_types = ['chaser', 'ambusher', 'random', 'chaser']  # Assign different types
    for i in range(4):
        ghost_pos = generate_ghost_position(pacman_start_pos, min_distance=120)
        if ghost_pos == pacman_start_pos:
            print(f"Ghost {i+1} could not find a safe spawn position.")
        ghost = Ghost(ghost_pos, ghost_type=ghost_types[i % len(ghost_types)])
        ghosts.add(ghost)
        all_sprites.add(ghost)
        print(f"Ghost {i+1} spawned at {ghost_pos} as {ghost.ghost_type}")  # Debugging line

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

    while True:
        dt = clock.tick(60)  # Delta time in milliseconds

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if not game_over:
            keys_pressed = pygame.key.get_pressed()
            pacman.update(keys_pressed)

            # Calculate average ghost distance
            distances = [pygame.math.Vector2(ghost.rect.center).distance_to(pacman.rect.center) for ghost in ghosts]
            average_ghost_distance = sum(distances) / len(distances) if distances else 150

            # Update difficulty
            difficulty_level = calculate_difficulty(current_score, current_lives, average_ghost_distance)

            # Optionally cap the difficulty to prevent ghosts from becoming too fast
            MAX_DIFFICULTY = 2.5
            if difficulty_level > MAX_DIFFICULTY:
                difficulty_level = MAX_DIFFICULTY

            # Update ghosts with new difficulty
            for ghost in ghosts:
                ghost.update(pacman.rect.center, pacman.direction, difficulty_level)

            # Handle invincibility timer
            if invincible:
                invincibility_timer -= dt
                if invincibility_timer <= 0:
                    invincible = False

            # Distance-Based Collision Detection
            collision = False
            for ghost in ghosts:
                distance = pygame.math.Vector2(pacman.rect.center).distance_to(ghost.rect.center)
                if distance < collision_threshold:
                    collision = True
                    break

            if collision and not invincible:
                current_lives -= 1
                print(f"Collision detected! Lives remaining: {current_lives}")  # Debugging line
                if current_lives <= 0:
                    print("Game Over!")
                    game_over = True
                else:
                    pacman.rect.center = pacman_start_pos  # Reset Pac-Man position
                    # Reposition ghosts safely
                    for i, ghost in enumerate(ghosts):
                        new_pos = generate_ghost_position(pacman_start_pos, min_distance=120)
                        if new_pos == pacman_start_pos:
                            print(f"Ghost {i+1} could not find a safe repositioning.")
                        ghost.rect.center = new_pos
                        print(f"Ghost {i+1} repositioned to {new_pos}")  # Debugging line
                    # Activate invincibility
                    invincible = True
                    invincibility_timer = INVINCIBILITY_DURATION

            # Update score (simple example: increase over time)
            current_score += 1

            # Render everything
            SCREEN.fill((0, 0, 0))
            all_sprites.draw(SCREEN)

            # Optionally, draw the minimum distance circle
            MIN_DISTANCE = 120
            pygame.draw.circle(SCREEN, (0, 0, 255), pacman.rect.center, MIN_DISTANCE, 1)

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
            SCREEN.blit(game_over_text, (SCREEN_WIDTH//2 - game_over_text.get_width()//2, SCREEN_HEIGHT//2 - 50))
            SCREEN.blit(final_score_text, (SCREEN_WIDTH//2 - final_score_text.get_width()//2, SCREEN_HEIGHT//2))
            pygame.display.flip()

            # Wait for a few seconds before quitting
            pygame.time.delay(3000)
            pygame.quit()
            sys.exit()

if __name__ == "__main__":
    main()
