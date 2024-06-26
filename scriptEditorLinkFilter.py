"""
This module creates clickable links from error messages in the maya script editor output window.

To use, call install().

Any errors reported in the script editor will be highlighted and become clickable links.
If the user clicks on one of these 'links' it will open the
user specified IDE to that file and line number.

Similarly you can call remove() to return the script editor to default.

Make sure you replace IDE_EXE and IDE_CMD with the exe and command for your own IDE.

To execute on maya startup, add the following to your userSetup.py.

EXAMPLE:
    from maya import cmds
    import scriptEditorLinkFilter
    cmds.evalDeferred("scriptEditorLinkFilter.install()", lowestPriority=True)

"""
import os
import re
import subprocess

from PySide6.QtWidgets import QWidget
from PySide6 import QtWidgets, QtCore, QtGui
from shiboken6 import wrapInstance

from maya import OpenMayaUI as omui


SCRIPT_EDITOR_OUTPUT_WIDGET_NAME = 'cmdScrollFieldReporter1'
TOGGLE_KEY = QtCore.Qt.Key.Key_Control
LINK_COLOR = (54, 84, 255, 255)
LINK_HOVER_COLOR = (120, 165, 255, 255)

# Regex to parse error lines
RE_PATTERN = r'(?P<file_name>(?<=File \").*)(?P<line_number>\", line \d*)'

# IDE executable path - replace with your own
IDE_EXE = r"C:\Program Files\JetBrains\PyCharm Community Edition 2023.1.2\bin\pycharm64.exe"

# IDE command to execute - replace this with the formatting it specifies
IDE_CMD = f'{IDE_EXE} --line {{line_num}} {{file_path}}'

global event_filter


def getScriptEditorOutputWidget():
    ptr = omui.MQtUtil.mainWindow()
    widget = wrapInstance(int(ptr), QWidget)
    ww = widget.findChild(QtWidgets.QPlainTextEdit, SCRIPT_EDITOR_OUTPUT_WIDGET_NAME)
    return ww


def install():
    global event_filter
    script_editor_wid = getScriptEditorOutputWidget()
    event_filter = ScriptLinkFilter(parent=script_editor_wid)
    event_filter.searchDocumentForLinks()
    return script_editor_wid, event_filter


def remove():
    global event_filter
    event_filter.doc.contentsChange.disconnect(event_filter.highlightNewBlocks)
    event_filter.restoreDocumentToDefault()
    event_filter.parent.removeEventFilter(event_filter)
    event_filter = None


class ScriptLinkFilter(QtCore.QObject):

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.parent = parent
        self.doc = self.parent.document()

        # Link formatting
        self.link_format = QtGui.QTextCharFormat()
        self.link_format.setForeground(QtGui.QColor.fromRgb(*LINK_COLOR))
        self.link_format.setFontUnderline(True)

        # Hovered link formatting
        self.link_hover_format = QtGui.QTextCharFormat()
        self.link_hover_format.setForeground(QtGui.QColor.fromRgb(*LINK_HOVER_COLOR))
        self.link_hover_format.setFontUnderline(True)

        # Default formatting
        self.default_format = QtGui.QTextCharFormat()

        self.cursor = None
        self.mouse_over_link = False
        self.stored_cursor_pos = 0
        self.stored_scroll_pos = [0, 0]
        self.highlighted_block = 0
        self.link_blocks = []

        # Connect signal and event filter
        self.doc.contentsChange.connect(self.highlightNewBlocks)
        self.parent.installEventFilter(self)

    @staticmethod
    def matchTextForFileAndLine(text):
        """
        Run the regex on text to match for file name and line number.

        Args:
            text:

        Returns:

        """
        match = re.search(RE_PATTERN, text)
        return match

    @staticmethod
    def pauseEvents(time):
        """
        Function to pause event processing.

        """
        # Create a QEventLoop
        loop = QtCore.QEventLoop()

        # Create a QTimer to resume event processing after time
        QtCore.QTimer.singleShot(time, loop.quit)

        # Process pending events until the timer expires
        loop.exec_()

    def highlightNewBlocks(self, position, charsRemoved, charsAdded):
        """
        Reads new text as it is added to the document and highlights
        error line blocks.

        """
        if charsAdded:
            self.doc.blockSignals(True)

            # Get the cursor and text under it
            cursor = self.parent.textCursor()
            cursor.setPosition(position)
            block = self.doc.findBlock(position)
            text = block.text()

            # Check for error lines
            match = self.matchTextForFileAndLine(text)

            if match:
                file_path = match['file_name']
                if os.path.exists(file_path):
                    self.setLineFormat(cursor, line_format=self.link_format)
                    self.link_blocks.append(block)

            self.doc.blockSignals(False)

        # If the doc is empty, reset vars.
        if self.doc.isEmpty():
            self.mouse_over_link = False
            self.stored_cursor_pos = 0
            self.stored_scroll_pos = [0, 0]
            self.highlighted_block = 0
            self.link_blocks = []

    def setCursorOverLink(self, cursor, line_format=None):
        """
        Store text block, set block format, and set
        the cursor to pointer hand.

        Args:
            cursor (QTextCursor):
            line_format (QTextCharFormat):

        Returns:

        """
        line_format = line_format or self.link_format
        self.highlighted_block = cursor.block()
        self.setLineFormat(cursor, line_format=line_format)
        self.stored_cursor_pos = cursor.position()
        self.mouse_over_link = True
        QtGui.QGuiApplication.setOverrideCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def setLineFormat(self, cursor, line_format=None):
        """
        Set the line format for the block under the cursor.

        Args:
            cursor (QTextCursor):
            line_format (QTextCharFormat):

        Returns:

        """
        self.doc.blockSignals(True)
        line_format = line_format or self.default_format

        pos = cursor.position()
        cursor.select(QtGui.QTextCursor.SelectionType.LineUnderCursor)
        cursor.setCharFormat(line_format)
        cursor.clearSelection()
        cursor.setPosition(pos)

        self.parent.setTextCursor(cursor)
        self.doc.blockSignals(False)

    def restoreDefaults(self, cursor, line_format=None):
        """
        Restore the default formatting for the block
        under the cursor.

        Args:
            cursor (QTextCursor):
            line_format (QTextCharFormat):

        Returns:

        """
        cursor.setPosition(self.stored_cursor_pos)
        self.setLineFormat(cursor, line_format=line_format)
        QtGui.QGuiApplication.restoreOverrideCursor()
        self.mouse_over_link = False
        self.highlighted_block = None

    def searchDocumentForLinks(self):
        """
        Scan the document text and find clickable links.

        Returns:

        """
        block = self.doc.firstBlock()
        num_blocks = self.doc.blockCount()
        cursor = self.parent.textCursor()
        cursor_pos = cursor.position()
        self.storeScrollBarPositions()

        i = 0
        while i < num_blocks:
            match = self.matchTextForFileAndLine(block.text())
            if match:
                if os.path.exists(match['file_name']):
                    cursor.setPosition(block.position())
                    self.setLineFormat(cursor, line_format=self.link_format)
                    self.link_blocks.append(block)
            block = block.next()
            i += 1

        cursor.setPosition(cursor_pos)
        self.restoreScrollBarPositions()

    def restoreDocumentToDefault(self):
        """
        Restore the document to default state.

        Returns:

        """
        cursor = self.parent.textCursor()
        self.storeScrollBarPositions()

        for block in self.link_blocks:
            cursor.setPosition(block.position())
            self.setLineFormat(cursor)

        self.restoreScrollBarPositions()
        QtGui.QGuiApplication.restoreOverrideCursor()

    def getTextUnderCursor(self):
        """
        Get the block text under the cursor.

        Returns:

        """
        cursor = self.parent.cursorForPosition(self.parent.mapFromGlobal(QtGui.QCursor().pos()))
        text = cursor.block().text()
        return cursor, text

    def storeScrollBarPositions(self):
        """
        Store positions of scroll bars.

        """
        self.stored_scroll_pos = [self.parent.horizontalScrollBar().value(),
                                  self.parent.verticalScrollBar().value()]

    def restoreScrollBarPositions(self):
        """
        Restore scroll bars to saved positions.

        """
        self.parent.horizontalScrollBar().setValue(self.stored_scroll_pos[0])
        self.parent.verticalScrollBar().setValue(self.stored_scroll_pos[1])

    def eventFilter(self, obj, event):

        # Mouse move
        # Highlights hovered links
        if event.type() in [QtCore.QEvent.Type.MouseMove, QtCore.QEvent.Type.TabletMove] \
                and event.buttons() == QtCore.Qt.MouseButton.NoButton:

            cursor_global_pos = QtGui.QCursor().pos()
            cursor_pos_local = self.parent.mapFromGlobal(cursor_global_pos)
            cursor = self.parent.cursorForPosition(cursor_pos_local)
            rect = self.parent.viewport().geometry()

            # If the mouse is over a link, highlight it
            if cursor.block() in self.link_blocks:
                if not self.mouse_over_link:
                    self.setCursorOverLink(cursor, line_format=self.link_hover_format)

            # This is the case for if we already are highlighted and
            # move off a link, return it to default link formatting.
            if self.mouse_over_link:

                # If we move onto scroll bars, we don't want to highlight.
                if not rect.contains(cursor_pos_local):
                    self.restoreDefaults(cursor, line_format=self.link_format)

                elif cursor.block() != self.highlighted_block:
                    self.restoreDefaults(cursor, line_format=self.link_format)

        # Extra security here, sometimes in the move events if you move
        # too fast it won't register as outside the rect before it stops
        # registering move events at all, so also use the leave event.
        if event.type() == QtCore.QEvent.Type.Leave:
            cursor_global_pos = QtGui.QCursor().pos()
            cursor_pos_local = self.parent.mapFromGlobal(cursor_global_pos)
            cursor = self.parent.cursorForPosition(cursor_pos_local)
            if self.mouse_over_link:
                self.restoreDefaults(cursor, line_format=self.link_format)

        # Mouse click
        # Launches editor to error file and line
        if event.type() in [QtCore.QEvent.Type.MouseButtonPress, QtCore.QEvent.Type.TabletPress]:
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                if self.mouse_over_link:
                    cursor, text = self.getTextUnderCursor()

                    # Parse line
                    match = self.matchTextForFileAndLine(text)
                    file_path = match['file_name']
                    line_num = match['line_number'].split()[-1]

                    # Launch the editor
                    cmd_string = IDE_CMD.format(line_num=line_num, file_path=file_path)
                    subprocess.Popen(cmd_string)

                    # Restore the document to default
                    # Necessary to pause events because otherwise any mouse move
                    # events would re-highlight the link, and it would be left in that state.
                    self.pauseEvents(10)
                    self.restoreDefaults(cursor, line_format=self.link_format)

                    return True

        return super().eventFilter(obj, event)