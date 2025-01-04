# area_selector.py
import tkinter as tk

def select_area():
    root = tk.Tk()
    root.attributes('-fullscreen', True)
    root.wait_visibility(root)
    root.attributes('-alpha', 0.3)  # Make window semi-transparent

    canvas = tk.Canvas(root, cursor="cross")
    canvas.pack(fill=tk.BOTH, expand=True)

    rect = None

    def on_press(event):
        nonlocal rect
        rect = canvas.create_rectangle(event.x, event.y, event.x, event.y, outline='red', width=2)

    def on_drag(event):
        nonlocal rect
        canvas.coords(rect, canvas.coords(rect)[0], canvas.coords(rect)[1], event.x, event.y)

    def on_release(event):
        root.quit()  # Exit window after selection

    canvas.bind('<ButtonPress-1>', on_press)
    canvas.bind('<B1-Motion>', on_drag)
    canvas.bind('<ButtonRelease-1>', on_release)

    root.mainloop()
    return canvas.coords(rect)  # Return the coordinates as (x1, y1, x2, y2)
