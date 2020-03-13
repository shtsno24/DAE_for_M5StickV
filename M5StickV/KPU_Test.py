import sensor, image, time, lcd
import KPU as kpu
import ulab as np

lcd.init(freq=24000000)
lcd.rotation(2)  # Rotate the lcd 180deg
sensor.reset()

sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.HQQQVGA)
sensor.set_windowing((32, 32))
sensor.skip_frames(100)
sensor.run(1)

print("init kpu")
lcd.draw_string(10, 10, "init kpu")
lcd.draw_string(170, 10, "Running")

lcd.draw_string(10, 30, "load kmodel")
kpu.memtest()
task = kpu.load(0x500000)
lcd.draw_string(170, 30, "Done")

lcd.draw_string(10, 50, "set outputs")
fmap = kpu.set_outputs(task, 0, 32, 32, 3)
kpu.memtest()
lcd.draw_string(170, 50, "Done")

print("Done")
time.sleep_ms(1000)
lcd.draw_string(170, 10, "Done     ")
time.sleep_ms(500)
lcd.draw_string(60, 70, "Setup Done! :)")
clock = time.clock()

img_object = image.Image()
color_list = [0, 0, 0]
scale = 3

while True:
    try:
        clock.tick()
        img = sensor.snapshot()         # Take a picture and return the image.
        fmap = kpu.forward(task, img)
        data_tuple = fmap[:]
        for i in range(1024):
            for c in range(3):
                data = data_tuple[c * 1024 + i]
                data *= 255
                if data < 0:
                    data = 0
                if data > 255:
                    data = 255
                color_list[c] = int(data)
            x = i % 32
            y = int(i / 32)
            color_tuple = (color_list[0], color_list[1], color_list[2])
            for _x in range(scale):
                for _y in range(scale):
                    a = img_object.set_pixel(scale * x + _x, scale * y + _y, color_tuple)

        fps = clock.fps()
        fps_0 = int(fps)
        fps_1 = (int(fps * 10) - fps_0 * 10)
        fps_2 = (int(fps * 100) - fps_1 * 10 - fps_0 * 100)
        fps_str = str(fps_0) + "." + str(fps_1) + str(fps_2) + " FPS"
        a = img_object.draw_string(120, 10, fps_str, color=(30, 111, 150), scale=2.5, mono_space=False)
        lcd.display(img_object)
        a = img_object.clear()

    except Exception as inst:
        print(inst)
        break

a = kpu.deinit(task)
lcd.clear((30, 111, 150))
