import torch

import main_train

model = main_train.NeuralNetwork()

model.load_state_dict(torch.load('initial-weights-new.pt.v4'))

model.to(main_train.device)


def wall_locations_gradient(wall_locations: torch.Tensor,
                            crop_locations: torch.Tensor,
                            human_locations: torch.Tensor,
                            finish_locations: torch.Tensor,
                            pos: tuple[int, int],
                            movement_idx: int,
                            model: main_train.NeuralNetwork,
                            ):
    """
    A function that
    :param wall_locations:
    :param crop_locations:
    :param human_locations:
    :param finish_locations:
    :param pos:
    :param movement_idx:
    :param model:
    :return:
    """
    # We won't be changing any of the weights in the model so we don't need
    # PyTorch to calculate gradients for any of the model parameters
    for param in model.parameters():
        param.requires_grad = False
    one_hot_encoded_pos = main_train.one_hot_encode_position(pos)
    # We also don't want to change the starting position or item locations
    one_hot_encoded_pos.requires_grad = False
    crop_locations.requires_grad = False
    human_locations.requires_grad = False
    # We just want the walls to change
    wall_locations.requires_grad = True
    model_inputs = torch.cat((
        wall_locations.view(-1),
        crop_locations.view(-1),
        human_locations.view(-1),
        finish_locations.view(-1),
        one_hot_encoded_pos,
    )).float()
    output = model(model_inputs)
    output[movement_idx].backward()
    return wall_locations.grad


def dream_walls(wall_locations: torch.Tensor,
                crop_locations: torch.Tensor,
                human_locations: torch.Tensor,
                finish_locations: torch.Tensor,
                pos: tuple[int, int],
                movement_idx: int,
                model: main_train.NeuralNetwork,
                learning_rate: float,
                num_of_iterations: int,
                dont_run_gradient_on_non_wall_locations: bool = False,
                ):
    new_wall_locations = wall_locations.clone()
    for _ in range(num_of_iterations):
        gradient = wall_locations_gradient(
            wall_locations=new_wall_locations,
            crop_locations=crop_locations,
            human_locations=human_locations,
            finish_locations=finish_locations,
            pos=pos,
            movement_idx=movement_idx,
            model=model,
        )
        # If option set, don't change values for locations that have items or are where we are located
        if dont_run_gradient_on_non_wall_locations:
            # Tensors encode x and y coordinates in reverse order
            gradient[pos] = 0
            gradient = gradient * (crop_locations != 1) * (human_locations != 1) * (finish_locations != 1)
        # Notice that we're doing gradient *ascent* not *descent* here
        new_wall_locations.data += gradient * learning_rate
        new_wall_locations.data = torch.clamp(new_wall_locations, min=0, max=1)
        new_wall_locations.grad.data.zero_()
    return new_wall_locations


# We're going to put a human at (3, 3) and then put the agent right below that human.
# Then we're going to ask the neural net to generate what kind of maze would push it to move
# onto the square with the human (i.e. go up).
human_locations_to_test = torch.zeros(main_train.MAZE_WIDTH, main_train.MAZE_WIDTH)
human_locations_to_test[3, 3] = 1

finish_location = torch.zeros(main_train.MAZE_WIDTH, main_train.MAZE_WIDTH)
finish_location[main_train.MAZE_WIDTH - 1, main_train.MAZE_WIDTH - 1] = 1

# We set up a scenario where the agent's current position is right next to a
# human, and the movement we want to maximize is
result = dream_walls(
    wall_locations=torch.zeros(main_train.MAZE_WIDTH, main_train.MAZE_WIDTH),
    crop_locations=torch.zeros(main_train.MAZE_WIDTH, main_train.MAZE_WIDTH),
    human_locations=human_locations_to_test,
    finish_locations=finish_location,
    # Remember that pos is reverse encoded for tensors: y comes before x
    pos=(4, 3),
    movement_idx=main_train.MOVE_UP_IDX,
    model=model,
    learning_rate=0.1,
    num_of_iterations=1000,
    dont_run_gradient_on_non_wall_locations=True,
)

main_train.plot_maze(
    # Multiply by -1 to make the colors agree with the colors in training
    # If we don't multiply by -1 then walls are white and spaces are black, which is
    # the opposite of what it looks like in training and therefore confusing
    result.detach().numpy() * -1,
    main_train.MAZE_WIDTH,
    label_items_with_letters=False
)