"""
This module creates clickable links from error messages in the maya script editor output window.

The event filter assigns a toggle key.  If that key is held, errors with a file and line number
will show up highlighted.  If the user clicks on one of these 'links' it will open the
user specified IDE to that file and line number.

To use, call install().

"""
import os
import re
import subprocess

from PySide6.QtWidgets import QWidget
from PySide6 import QtWidgets, QtCore, QtGui
from shiboken6 import wrapInstance

from maya import OpenMayaUI as omui

SCRIPT_EDITOR_WIDGET_NAME = 'cmdScrollFieldReporter1'
TOGGLE_KEY = QtCore.Qt.Key.Key_Control
LINK_COLOR = (54, 84, 255, 255)
LINK_HOVER_COLOR = (120, 165, 255, 255)

# Regex to parse error lines
RE_PATTERN = r'(?P<file_name>(?<=File \").*)(?P<line_number>\", line \d*)'

# IDE executable path
IDE_EXE = r"C:\Program Files\JetBrains\PyCharm Community Edition 2023.1.2\bin\pycharm64.exe"


def getScriptEditorOutputWidget():
    ptr = omui.MQtUtil.mainWindow()
    widget = wrapInstance(int(ptr), QWidget)
    ww = widget.findChild(QtWidgets.QPlainTextEdit, SCRIPT_EDITOR_WIDGET_NAME)
    return ww


def install():
    script_editor_wid = getScriptEditorOutputWidget()
    event_filter = ScriptLinkFilter()
    script_editor_wid.installEventFilter(event_filter)
    return script_editor_wid, event_filter


class ScriptLinkFilter(QtCore.QObject):

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # Link formatting
        self.link_format = QtGui.QTextCharFormat()
        self.link_format.setForeground(QtGui.QColor.fromRgb(*LINK_COLOR))
        self.link_format.setFontUnderline(True)

        # Hovered link formatting
        self.link_hover_format = QtGui.QTextCharFormat()
        self.link_hover_format.setForeground(QtGui.QColor.fromRgb(*LINK_HOVER_COLOR))
        self.link_hover_format.setFontUnderline(True)

        self.default_format = QtGui.QTextCharFormat()

        self.obj = None
        self.key_held = False
        self.mouse_over_link = False
        self.stored_cursor_pos = 0
        self.highlighted_block = 0
        self.link_blocks = []

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

    def setCursorOverLink(self, cursor, line_format=None):
        """
        Store text block, set block format, and set
        the cursor to pointer hand.

        Args:
            cursor (QTextCursor):
            line_format:

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
            cursor:
            line_format:

        Returns:

        """

        line_format = line_format or self.default_format
        cursor.select(QtGui.QTextCursor.SelectionType.LineUnderCursor)
        cursor.setCharFormat(line_format)
        cursor.clearSelection()
        self.obj.setTextCursor(cursor)

    def restoreDefaults(self, cursor, line_format=None):
        """
        Restore the default formatting for the block
        under the cursor.

        Args:
            cursor:
            line_format:

        Returns:

        """
        cursor.setPosition(self.stored_cursor_pos)
        self.setLineFormat(cursor, line_format=line_format)
        QtGui.QGuiApplication.restoreOverrideCursor()
        self.mouse_over_link = False
        self.highlighted_block = None

    def getTextUnderCursor(self):
        """
        Get the block text under the cursor.

        Returns:

        """
        cursor = self.obj.cursorForPosition(self.obj.mapFromGlobal(QtGui.QCursor().pos()))
        text = cursor.block().text()
        return cursor, text

    def searchDocumentForLinks(self):
        """
        Scan the document text and find clickable links.

        Returns:

        """
        doc = self.obj.document()
        block = doc.firstBlock()
        num_blocks = doc.blockCount()
        i = 0
        while i < num_blocks:
            match = self.matchTextForFileAndLine(block.text())
            if match:
                if os.path.exists(match['file_name']):
                    self.link_blocks.append(block)
            block = block.next()
            i += 1

    def restoreDocumentToDefault(self):
        """
        Restore the document to default state.

        Returns:

        """
        cursor = self.obj.textCursor()
        for block in self.link_blocks:
            cursor.setPosition(block.position())
            self.setLineFormat(cursor)
        QtGui.QGuiApplication.restoreOverrideCursor()
        self.link_blocks = []
        self.mouse_over_link = False

    def eventFilter(self, obj, event):

        self.obj = obj

        # Toggle key press
        # Scans document for clickable links and highlights them
        if event.type() == QtCore.QEvent.Type.KeyPress and event.key() == TOGGLE_KEY:
            self.searchDocumentForLinks()
            cursor = obj.textCursor()

            # Move the cursor to each block and highlight them
            for block in self.link_blocks:
                cursor.setPosition(block.position())
                self.setLineFormat(cursor, line_format=self.link_format)

            self.key_held = True

        # Toggle key release
        # Restores the formatting to default
        if event.type() == QtCore.QEvent.Type.KeyRelease and event.key() == TOGGLE_KEY:
            self.restoreDocumentToDefault()
            self.key_held = False

        # Mouse move
        # Highlights hovered links
        if event.type() in [QtCore.QEvent.Type.MouseMove, QtCore.QEvent.Type.TabletMove]:
            if self.key_held:
                cursor = self.obj.cursorForPosition(self.obj.mapFromGlobal(QtGui.QCursor().pos()))

                # If the mouse is over a link, highlight it
                if cursor.block() in self.link_blocks:
                    if not self.mouse_over_link:
                        self.setCursorOverLink(cursor, line_format=self.link_hover_format)

                # This is the case for if we already are highlighted and
                # move off a link, return it to default link formatting.
                if self.mouse_over_link:
                    if cursor.block() != self.highlighted_block:
                        self.restoreDefaults(cursor, line_format=self.link_format)

        # Mouse click
        # Launches editor to error file and line
        if event.type() in [QtCore.QEvent.Type.MouseButtonPress, QtCore.QEvent.Type.TabletPress]:
            if self.mouse_over_link:
                cursor, text = self.getTextUnderCursor()

                # Parse line
                match = self.matchTextForFileAndLine(text)
                file_path = match['file_name']
                line_num = match['line_number'].split()[-1]

                # Launch the editor
                cmd_string = f'{IDE_EXE} --line {line_num} {file_path}'
                subprocess.Popen(cmd_string)

                # Restore the document to default
                self.restoreDocumentToDefault()

                return True

        return super().eventFilter(obj, event)


