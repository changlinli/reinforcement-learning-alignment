import random

def generate_maze(width, height):
    # Directions for moving up, down, left, right
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # Initialize maze with walls
    maze = [[1 for _ in range(2*width+1)] for _ in range(2*height+1)]
    for x in range(width):
        for y in range(height):
            maze[2*y+1][2*x+1] = 0
    
    # Initialize stack and starting cell
    stack = []
    x, y = random.randint(0, width-1), random.randint(0, height-1)
    stack.append((x, y))
    
    # Mark starting cell as visited
    maze[2*y+1][2*x+1] = 0
    
    # Helper function to check if cell is valid and unvisited
    def is_valid(nx, ny):
        return 0 <= nx < width and 0 <= ny < height and maze[2*ny+1][2*nx+1] == 1
    
    while stack:
        x, y = stack[-1]
        # Get unvisited neighbors
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny):
                neighbors.append((nx, ny))
        
        if neighbors:
            # Choose a random unvisited neighbor
            nx, ny = random.choice(neighbors)
            
            # Remove wall between current cell and chosen cell
            maze[y + dy][x + dx] = 0
            
            # Mark chosen cell as visited
            maze[2*ny+1][2*nx+1] = 0
            
            # Push chosen cell to the stack
            stack.append((nx, ny))
        else:
            # Pop from stack
            stack.pop()
    
    return maze

# Generate and display maze
maze = generate_maze(10, 10)
for row in maze:
    print("".join(['#' if cell else ' ' for cell in row]))
