"""Run DeskGuard's chat in a local window without an HTTP listener."""

import re
import tkinter as tk
from concurrent.futures import Future
from threading import Thread
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from chat import DeskGuard, QUESTION_MAX_LENGTH

BACKGROUND = "#f5f6f2"
INK = "#172e2b"
ACCENT = "#145548"
MUTED = "#4f625d"


class DesktopChat:
    def __init__(self, window: tk.Tk , user_id : int = 10 ):
        self.chat = DeskGuard()
        self.window = window
        self.last_reply = None
        window.title("DeskGuard — Live Chat")
        window.geometry("830x520+36+32")
        window.minsize(560, 410)
        window.configure(background=BACKGROUND)

        body = tk.Frame(window, background=BACKGROUND, padx=24, pady=18)
        body.pack(fill="both", expand=True)
        tk.Label(body, text="DeskGuard", background=BACKGROUND, foreground=INK,
                 font=("Segoe UI", 24, "bold")).pack(anchor="w")
        mode = "Model answers" if self.chat.model_enabled else "Local passage preview"
        tk.Label(body, text=f"Fictional banking support  ·  {mode}", background=BACKGROUND,
                 foreground=MUTED, font=("Segoe UI", 10)).pack(anchor="w", pady=(0, 12))

        self.transcript = ScrolledText(
            body, width=1, height=1, wrap="word", font=("Segoe UI", 11), background="white",
            foreground=INK, relief="flat", padx=16, pady=12, state="disabled",
        )
        self.transcript.pack(fill="both", expand=True, pady=(0, 12))
        self.transcript.tag_configure("speaker", foreground=ACCENT, font=("Segoe UI", 11, "bold"))
        self.append("DeskGuard", "Ask about cards, transfers, UPI, refunds, sign-in, or account settings.")

        composer = ttk.Frame(body)
        composer.pack(fill="x")
        self.question = ttk.Entry(composer, font=("Segoe UI", 12))
        self.question.pack(side="left", fill="x", expand=True, ipady=4)
        self.question.bind("<Return>", self.send)
        self.send_button = ttk.Button(composer, text="Send", command=self.send)
        self.send_button.pack(side="left", padx=(8, 0), ipady=4)

        footer = tk.Frame(body, background=BACKGROUND)
        footer.pack(fill="x", pady=(8, 0))
        self.inspect_button = ttk.Button(
            footer, text="Retrieved passages", command=self.inspect, state="disabled"
        )
        self.inspect_button.pack(side="left")
        self.status = tk.StringVar(value="Ask a question to see sources and matching passages.")
        tk.Label(footer, textvariable=self.status, background=BACKGROUND,
                 foreground=MUTED, font=("Segoe UI", 9)).pack(side="right")
        self.question.focus_set()

    def append(self, speaker, text):
        self.append_text(f"{speaker}\n", "speaker")
        self.append_text(f"{text}\n\n")

    def append_text(self, text, *tags, follow=True):
        self.transcript.configure(state="normal")
        self.transcript.insert("end", text, tags)
        self.transcript.configure(state="disabled")
        if follow:
            self.transcript.see("end")

    def send(self, _event=None):
        question = self.question.get().strip()
        if self.send_button.instate(("disabled",)) or not question:
            return
        if len(question) > QUESTION_MAX_LENGTH:
            self.status.set(f"Questions can contain up to {QUESTION_MAX_LENGTH:,} characters.")
            return

        self.question.delete(0, "end")
        self.send_button.configure(state="disabled")
        self.inspect_button.configure(state="disabled")
        self.append("You", question)
        self.status.set("Finding relevant passages…")

        result = Future()

        def request_reply():
            try:
                result.set_result(self.chat.ask(question))
            except Exception as error:
                result.set_exception(error)

        Thread(target=request_reply, daemon=True).start()
        self.window.after(50, self.show_reply, result)

    def show_reply(self, result):
        if not result.done():
            self.window.after(50, self.show_reply, result)
            return

        try:
            reply = result.result()
        except Exception:
            self.last_reply = None
            self.append("DeskGuard", "I couldn’t process that question. Please try again.")
            self.status.set("Ready")
            self.send_button.configure(state="normal")
            self.question.focus_set()
            return

        label = "DeskGuard · Article preview" if reply["status"] == "preview" else "DeskGuard"
        self.append_text(f"{label}\n", "speaker")
        self.status.set("Writing reply.")
        words = re.findall(r"\S+\s*|\s+", reply["answer"])
        delay = max(45, min(85, 7000 // max(1, len(words))))
        word_iter = iter(words)

        def type_next_word():
            follow = self.transcript.yview()[1] >= 0.999
            word = next(word_iter, None)
            if word is None:
                self.finish_reply(reply, follow)
                return
            self.append_text(word, follow=follow)
            self.window.after(delay, type_next_word)

        type_next_word()

    def finish_reply(self, reply, follow):
        suffix = "\n\n"
        if reply["sources"]:
            sources = [f"• {s['title']} — {s['section']}" for s in reply["sources"]]
            suffix += "Sources\n" + "\n".join(sources) + "\n\n"
        self.append_text(suffix, follow=follow)
        self.last_reply = reply
        self.inspect_button.configure(state="normal")
        self.status.set(f"Retrieved passages for the last reply: {len(reply['retrieved'])}")
        self.send_button.configure(state="normal")
        self.question.focus_set()

    def inspect(self):
        reply = self.last_reply
        if reply is None:
            return
        dialog = tk.Toplevel(self.window)
        dialog.title("DeskGuard — Retrieved passages")
        dialog.geometry("720x520")
        content = ScrolledText(dialog, wrap="word", font=("Segoe UI", 10), padx=16, pady=12)
        content.pack(fill="both", expand=True)
        if not reply["retrieved"]:
            content.insert("end", "No passages matched this question.\n")
        for item in reply["retrieved"]:
            chunk = item["chunk"]
            content.insert(
                "end",
                f"{chunk['title']} — {chunk['section']}\n"
                f"{chunk['id']} · cosine similarity {item['similarity']:.3f}\n"
                f"{chunk['text']}\n\n",
            )
        label = "Model input" if reply["model_requested"] else "Prepared model input"
        model = reply["model_input"]
        content.insert("end", f"\n{label}\n\n{model['instructions']}\n\n{model['input']}")
        content.configure(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    DesktopChat(root)
    root.mainloop()
