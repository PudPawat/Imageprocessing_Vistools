
import sys
import pygame
import cv2
from pygame.locals import KEYDOWN, K_ESCAPE, K_q

def start(screen):
    camera = cv2.VideoCapture(0)
    pygame.init()
    pygame.display.set_caption("OpenCV camera stream on Pygame")
    # screen = pygame.display.set_mode([640, 480])
    while True:
        ret, frame = camera.read()
        frame = cv2.resize(frame,(500,500))

        screen.fill([0, 0, 0])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.swapaxes(0, 1)
        pygame.surfarray.blit_array(screen, frame)
        pygame.display.update()

        control = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
                # sys.exit(0)
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE or event.key == K_q:
                    print("outttt!!!!")
                    control = 1
                    break
                    # sys.exit(0)
            # else:
            #     continue
        if control == 1:
            break


def Button(screen, position, text, button_size=(200, 50)):
    left, top = position
    bwidth, bheight = button_size
    pygame.draw.line(screen, (150, 150, 150), (left, top), (left+bwidth,top),5)
    pygame.draw.line(screen, (150, 150, 150), (left, top-2), (left, top+bheight), 5)
    pygame.draw.line(screen, (50, 50, 50), (left, top+bheight), (left+bwidth, top+bheight), 5)
    pygame.draw.line(screen, (50, 50, 50), (left+bwidth, top+bheight), (left+bwidth, top), 5)
    pygame.draw.rect(screen, (100, 100, 100), (left, top, bwidth, bheight))
    font = pygame.font.Font("font.TTF", 30)
    text_render = font.render(text, 1, (255, 235, 205))
    return screen.blit(text_render, (left+50, top+10))

def startInterface(screen):
    clock = pygame.time.Clock()
    while True:
        screen.fill((41, 36, 33))
        button_1 = Button(screen, (150, 175), 'Setting')
        button_2 = Button(screen, (150, 275), 'Start')
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if button_1.collidepoint(pygame.mouse.get_pos()):
                    return 1
                elif button_2.collidepoint(pygame.mouse.get_pos()):
                    return 2
        clock.tick(10)
        pygame.display.update()


def runDemo(screen):
    game_mode = startInterface(screen)
    while True:
        for event in pygame.event.get():
            print(event.type)
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    print("asdasd")
                    # pygame.quit()
                    # sys.exit(-1)
                    break
            # elif event.key == K_ESCAPE:
            #     pygame.display.update()
            #     break
        screen.fill((41, 36, 33))
        if game_mode ==  1 :
            screen.fill((0,0,0))
            pygame.display.update()
            game_mode = startInterface(screen)
        elif game_mode == 2:
            start(screen)
            game_mode = startInterface(screen)
            pygame.display.update()
        pygame.display.update()


def endInterface(screen):
    while True:
        screen.fill((41, 36, 33))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    return
                elif event.key == pygame.K_ESCAPE:
                    sys.exit()
                    pygame.quit()
        for text, pos in zip(texts, positions):
            screen.blit(text, pos)
        clock.tick(10)
        pygame.display.update()


def main():
    pygame.init()
    pygame.mixer.init()
    screen = pygame.display.set_mode((500, 500))
    pygame.display.set_caption('pingpong - 微信公众号: Charles的皮卡丘')
    # 开始游戏
    while True:
        runDemo(screen)
        # endInterface(screen)

if __name__ == '__main__':
    main()