import pyatspi



def find_application_by_name(desktop, app_name):
    for app in desktop:
        if app.name.lower() == app_name.lower():
            return app
    return None
# Get the desktop object
desktop = pyatspi.Registry.getDesktop(0)

# You would need to write code to navigate through the accessible objects
# For example, to list all the accessible applications:
for app in desktop:
    for i in range(0, app.childCount):
        child = app.getChildAtIndex(i)
        print(child)
    
# Function to perform a click action on an application
def click_application(app):
    # Find the 'click' action
    print('#####')
    print(app.childCount)
    for i in range(0, app.childCount):
        child = app.getChildAtIndex(i)
        print(child)
        action = child.queryAction()
        for j in range(0, action.nActions):
            if action.getName(j) in ['click', 'activate']:  # Action names can vary
                action.doAction(j)
                return True
    return False

# Get the desktop object
desktop = pyatspi.Registry.getDesktop(0)

# Find the Astrill application
astrill_app = find_application_by_name(desktop, "astrill")

# If found, perform the click action
if astrill_app:
    print(astrill_app)
    clicked = click_application(astrill_app)
    if clicked:
        print("Clicked on Astrill application.")
    else:
        print("Could not click on Astrill application.")
else:
    print("Astrill application not found.")