import pygame, sys
from pygame.locals import *
# following package is available at:
# https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter08/maze.py
import VariousMaze
import numpy as np

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BRIGHTBLUE = (0, 50, 255)
DARKTURQUOISE = (3, 54, 73)
GREEN = (0, 204, 0)
RED = (255, 0, 0)


def draw_state(dynaParams, state, step, current_episode):
    DISPLAYSURF.fill(BLACK)
    BASICFONT = pygame.font.Font('freesansbold.ttf', 30)
    state_text = "Running... " if state == RUNNING else "Pause   "
    textSurf = BASICFONT.render("{state_text}Planning step: {planning_step}   Episode: {current_episode}"
                                "     Step: {step}".
                                format(state_text=state_text, current_episode=current_episode, step=step,
                                       planning_step=dynaParams.planningSteps), True, WHITE)
    textRect = textSurf.get_rect()
    textRect.topleft = (10, 10)
    DISPLAYSURF.blit(textSurf, textRect)

def dynaQ_step(dynaParams, maze, current_state):
    # get action
    model = VariousMaze.TrivialModel()
    action = VariousMaze.chooseAction(current_state, maze.stateActionValues, maze, dynaParams)

    # take action
    newState, reward = maze.takeAction(current_state, action)

    # Q-Learning update
    maze.stateActionValues[current_state[0], current_state[1], action] += \
        dynaParams.alpha * (reward + dynaParams.gamma * np.max(maze.stateActionValues[newState[0], newState[1], :]) -
                            maze.stateActionValues[current_state[0], current_state[1], action])

    # feed the model with experience
    model.feed(current_state, action, newState, reward)

    # sample experience from the model
    for t in range(0, dynaParams.planningSteps):
        state_sample, action_sample, new_state_sample, reward_sample = model.sample()
        maze.stateActionValues[state_sample[0], state_sample[1], action_sample] += \
            dynaParams.alpha * (reward_sample + dynaParams.gamma * np.max(
                maze.stateActionValues[new_state_sample[0], new_state_sample[1], :]) -
                                maze.stateActionValues[state_sample[0], state_sample[1], action_sample])

    return newState



def draw_state_action_value(maze):
    top = 100
    left = 500
    width = 40
    height = 40
    color = WHITE
    gap = 2
    value_max = np.max(maze.stateActionValues)
    value_min = np.min(maze.stateActionValues)

    for i in range(maze.WORLD_HEIGHT):
        for j in range(maze.WORLD_WIDTH):
            for a in range(len(maze.actions)):
                # color = np.random.random((3)) * 255
                # color = color.astype(int)
                color = WHITE if value_min == value_max \
                    else (255 - int((maze.stateActionValues[i][j][a] - value_min) / (value_max - value_min) * 255), 255, 255)
                left_x = left + j * (width + gap)
                right_x = left_x + width
                top_y = top + i * (height + gap)
                bottom_y = top_y + height
                if a == maze.ACTION_UP:
                    pygame.draw.polygon(DISPLAYSURF, tuple(color), [(left_x, top_y), (right_x, top_y),
                                                                    (int((left_x + right_x) / 2),
                                                                     top_y + int(height / 2))])
                elif a == maze.ACTION_DOWN:
                    pygame.draw.polygon(DISPLAYSURF, color, [(left_x, bottom_y), (right_x, bottom_y),
                                                             (int((left_x + right_x) / 2), bottom_y - int(height / 2))])
                elif a == maze.ACTION_LEFT:
                    pygame.draw.polygon(DISPLAYSURF, color, [(left_x, top_y), (left_x, bottom_y),
                                                             (left_x + int(width / 2), top_y + int(height / 2))])
                else:
                    pygame.draw.polygon(DISPLAYSURF, color, [(right_x, top_y), (right_x, bottom_y),
                                                             (right_x - int(width / 2), top_y + int(height / 2))])


def draw_map(maze, current_state):
    top = 100
    left = 20
    width = 40
    height = 40
    # color = WHITE
    gap = 1
    for i in range(maze.WORLD_HEIGHT):
        for j in range(maze.WORLD_WIDTH):
            if current_state[0] == i and current_state[1] == j:
                color = DARKTURQUOISE
            elif maze.START_STATE[0] == i and maze.START_STATE[1] == j:
                color = GREEN
            elif [i, j] in maze.obstacles:
                color = BRIGHTBLUE
            elif maze.GOAL_STATES[0][0] == i and maze.GOAL_STATES[0][1] == j:
                color = RED
            else:
                color = WHITE
            pygame.draw.rect(DISPLAYSURF, color, (left + j * (width + gap), top + i * (height + gap), width, height))


def get_selected_planning_step(top, left, offset, button_height, button_width, mouse_x, mouse_y, planning_step_list):
    for index in range(len(planning_step_list)):
        if left + index * offset < mouse_x < left + index * offset + button_width and \
                top < mouse_y < top + button_height:
            return planning_step_list[index]

    return None


def draw_planning_steps(mousex=0, mousey=0):
    global DISPLAYSURF
    global selected_planning_step
    global state

    BASICFONT = pygame.font.Font('freesansbold.ttf', 20)
    textSurf = BASICFONT.render("Planning Step", True, (255, 255, 255))
    textRect = textSurf.get_rect()
    textRect.topleft = (30, 30)
    DISPLAYSURF.blit(textSurf, textRect)

    top = 20
    left = 200
    button_width = 50
    button_height = 50
    offset = 70
    i = 0
    planning_step_list = [0, 5, 50]

    if state == RUNNING:
        clicked_planning_step = get_selected_planning_step(top, left, offset, button_height, button_width, mouse_x, mouse_y,
                                                           planning_step_list)
        if clicked_planning_step is not None:
            selected_planning_step = clicked_planning_step

    for planning_step in planning_step_list:
        button_color = GREEN if selected_planning_step != planning_step else RED
        left_pos = left + offset * i
        pygame.draw.rect(DISPLAYSURF, button_color, (left_pos, top, button_width, button_height))
        textSurf = BASICFONT.render(str(planning_step), True, BLACK)
        textRect = textSurf.get_rect()
        textRect.center = left_pos + int(button_width / 2), top + int(button_height / 2)
        DISPLAYSURF.blit(textSurf, textRect)
        i += 1

pygame.init()

FPS = 30  # frames per second setting
fpsClock = pygame.time.Clock()

# set up the window
DISPLAYSURF = pygame.display.set_mode((1600, 1000), 0, 32)
pygame.display.set_caption('maze demo')

maze = VariousMaze.Maze()
dynaParams = VariousMaze.DynaParams()
dynaParams.planningSteps = 50
RUNNING, PAUSE = 0, 1
planning_step = 0
episode = 50
mouse_x = 0
mouse_y = 0
selected_planning_step = 0
current_episode = 1
current_state = maze.START_STATE
state = PAUSE
step = 0
while True:
    # draw_planning_steps(mouse_x, mouse_y)
    draw_state(dynaParams, state, step, current_episode)
    draw_map(maze, current_state)
    draw_state_action_value(maze)
    if state == RUNNING:
        current_state = dynaQ_step(dynaParams, maze, current_state)
        step += 1
        # print('current_state: ')
        # print(current_state)

        # if current_state[0] == maze.GOAL_STATES[0] and current_state[1] == maze.GOAL_STATES[1]:
        if current_state == maze.GOAL_STATES[0]:
            current_state = maze.START_STATE
            step = 0
            current_episode += 1

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == MOUSEBUTTONUP:
            mouse_x, mouse_y = event.pos
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                state = PAUSE
            if event.key == pygame.K_s:
                state = RUNNING

    pygame.display.update()
    fpsClock.tick(FPS)
