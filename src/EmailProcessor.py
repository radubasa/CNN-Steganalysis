import email
import imaplib
import os
from email.policy import default
from PIL import Image
import numpy as np

class EmailProcessor:
    def __init__(self, email_address, password, imap_server, imap_port=993):
        self.email_address = email_address
        self.password = password
        self.imap_server = imap_server
        self.imap_port = imap_port
        self.mail = imaplib.IMAP4_SSL(self.imap_server, self.imap_port)
        self.mail.login(self.email_address, self.password)

    def fetch_emails(self, folder='inbox'):
        self.mail.select(folder)
        result, data = self.mail.search(None, 'ALL')
        email_ids = data[0].split()
        return email_ids

    def extract_images_from_email(self, email_id):
        result, data = self.mail.fetch(email_id, '(RFC822)')
        raw_email = data[0][1]
        msg = email.message_from_bytes(raw_email, policy=default)
        images = []
        for part in msg.iter_attachments():
            if part.get_content_maintype() == 'image':
                image_data = part.get_payload(decode=True)
                image = Image.open(io.BytesIO(image_data))
                images.append(image)
        return images