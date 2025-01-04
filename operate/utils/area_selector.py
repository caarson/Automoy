import tkinter as tk

def select_area():
    try:
        # Main window for selection background
        root = tk.Tk()
        root.title("Background Overlay")
        root.attributes('-fullscreen', True)
        root.attributes('-alpha', 0.3)  # Semi-transparent
        root.configure(background='white')

        print("Main window configured")

        # Separate Toplevel window for drawing the rectangle
        outline_window = tk.Toplevel(root, bg='systemTransparent')
        outline_window.attributes('-alpha', 1)  # Fully opaque for visibility
        outline_window.attributes('-topmost', True)  # Keep this window on top
        outline_window.overrideredirect(True)  # Remove window decorations
        outline_canvas = tk.Canvas(outline_window, cursor="cross", bg='systemTransparent', bd=0, highlightthickness=0)
        outline_canvas.pack()

        print("Outline window configured")

        rect = None  # Store the rectangle

        def on_press(event):
            nonlocal rect
            # Initialize the rectangle
            rect = outline_canvas.create_rectangle(event.x_root, event.y_root, event.x_root, event.y_root, outline='red', width=4)
            print("Rectangle started at:", event.x_root, event.y_root)

        def on_drag(event):
            if rect:
                # Resize the rectangle as the mouse drags
                outline_canvas.coords(rect, outline_canvas.coords(rect)[0], outline_canvas.coords(rect)[1], event.x_root, event.y_root)
                print("Dragging to:", event.x_root, event.y_root)

        def on_release(event):
            # Once done, resize the outline window to just fit the rectangle
            coords = outline_canvas.coords(rect)
            outline_window.geometry(f"{int(coords[2] - coords[0])}x{int(coords[3] - coords[1])}+{int(coords[0])}+{int(coords[1])}")
            print("Rectangle finalized at:", coords)
            root.withdraw()  # Optionally hide the main window

        # Bind the events
        root.bind('<ButtonPress-1>', on_press)
        root.bind('<B1-Motion>', on_drag)
        root.bind('<ButtonRelease-1>', on_release)

        root.mainloop()
        return outline_canvas.coords(rect) if rect else None

    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    print("Starting application...")
    selected_area = select_area()
    print(f"Selected area coordinates: {selected_area}")
