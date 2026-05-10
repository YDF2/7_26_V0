# Help document for `\src`

<!--
Author:     Yifan Zhang
Date:       2019-7-1
Updated:    2019-7-2
Description:A document that helps programmers understanding the parts of this repo, however it mostly acts as a personnal note.
-->

This document along with comments in this repo is written while I'm reviewing the repo. Mistakes may have been made.

Comments start with a '?' mean that I'm not sure if my understanding is correct.

## `> controller`

*maxwell* responds to "止于至善"🙃.

### `>> drivers`

#### `>>> dynamixel`

Files here are for steers from *Dynamixel*.

#### `>>> gpio`

***

### `>> options`

Files here realize **program_options**. For example, in `./a.out -help`, `-help` is a **program_option**.

***

### `>> player`

Most of this part are quite literally.

>   *gc* stands for *gamecontrol*. Gamecontrol, provided by the organizing committee, sends signals to robots telling them if they are about to be leave the game, they have goaled, or nothing happend.

#### `>>> core`

#### `>>> engine`

Files here encapsulate functions that control motions of the robot.

>   *IK* stands for *Inverse Kinematics*.

>   [*Leph*](https://github.com/Leph) is a user on Github, while [*Rhoban*](https://github.com/Rhoban) is an organization with Leph inside.

#### `>>> fsm`

>   *FSM* stands for *Finite State Machine*. 

>   *SL* stands for *Self Localization*.

#### `>>> localization`

*inverse* and *kalman* (short for *kalman filter*) involve matrix calculation, which I am pretty weak at.

Notice: It is said that the localization is realized mostly via *particle filter* instead of *kalman filter*, however I haven't found any evidence to support this point yet.

#### `>>> sensor`

>   *Imu* stands for *Inertial measurement unit*.

#### `>>> skill`

Skills are combinations of serial motions.

#### `>>> task`

Tasks are also combinations of serial actions, but skills tend to be mechanical while tasks are not.

#### `>>> vision`

>   *spf* stands for *support foot*.

***

## `> data`

Files here store configurations of the robot.

## `> lib`

### `>> darknet`

Lib for recognition.

### `>> imageproc`

### `>> math`

### `>> parser`

### `>> robot`

### `>> udp_data`

## `> scripts`

[What does if __name__ == “__main__”: do?](https://stackoverflow.com/questions/419163/what-does-if-name-main-do) This make sure that the file is run as the main file, which means that it's run directly instead of being called.