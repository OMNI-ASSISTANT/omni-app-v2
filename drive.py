import os.path
import json # Though not directly used in SDK calls, good for token.json if manual inspection is needed.
import io # Added

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload # Added

# If modifying these scopes, delete token.json.
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"] # Updated scope
CLIENT_SECRET_FILE = "client_secret.json"
TOKEN_FILE = "token.json"

class DriveService:
    def authenticate_google_drive():
        """Shows basic usage of the Drive v3 API.
        Handles OAuth 2.0 authentication and returns an authenticated service object.
        """
        creds = None
        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists(TOKEN_FILE):
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    print(f"Failed to refresh token: {e}")
                    creds = None # Force re-authentication
            if not creds: # creds might be None if refresh failed or wasn't possible
                if not os.path.exists(CLIENT_SECRET_FILE):
                    print(f"Error: {CLIENT_SECRET_FILE} not found. Please download it from Google Cloud Console and place it in the same directory as this script.")
                    return None
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
                    # Add prompt='consent' to force re-authentication and refresh token acquisition if needed for testing
                    # creds = flow.run_local_server(port=0, prompt='consent') 
                    creds = flow.run_local_server(port=0)
                except Exception as e:
                    print(f"Error during authentication flow: {e}")
                    return None
            # Save the credentials for the next run
            try:
                with open(TOKEN_FILE, "w") as token:
                    token.write(creds.to_json())
            except Exception as e:
                print(f"Error saving token to {TOKEN_FILE}: {e}")
                # Continue without saving if there's an error, but warn user
                
        try:
            service = build("drive", "v3", credentials=creds)
            return service
        except HttpError as error:
            print(f"An error occurred building the service: {error}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    def search_drive_files(service, search_term: str) -> list:
        """
        Searches for files in Google Drive based on a search term using the SDK
        and returns a list of dictionaries, each containing the file's name, ID, and MIME type.
        Handles pagination.

        Args:
            service: Authenticated Google Drive API service object.
            search_term: The term to search for in file content, name, or description.

        Returns:
            A list of dictionaries, where each dictionary has 'name', 'id', and 'mimeType' keys.
            Returns an empty list if no files are found or if an error occurs.
        """
        results = []
        page_token = None
        escaped_search_term = search_term.replace("'", "\\'")
        query = f"fullText contains '{escaped_search_term}'"

        try:
            while True:
                response = (
                    service.files()
                    .list(
                        q=query,
                        spaces="drive", # Search in the user's My Drive
                        fields="nextPageToken, files(id, name, mimeType)",
                        pageToken=page_token,
                    )
                    .execute()
                )
                
                for file_item in response.get("files", []):
                    results.append({
                        "name": file_item.get("name"),
                        "id": file_item.get("id"),
                        "mimeType": file_item.get("mimeType")
                    })
                
                page_token = response.get("nextPageToken", None)
                if page_token is None:
                    break
                    
        except HttpError as error:
            print(f"An error occurred during the API request: {error}")
        except Exception as e: # Catch other potential errors
            print(f"An unexpected error occurred during search: {e}")
            
        return results

    # New function to download files
    def download_file_by_id(service, file_id: str, download_path: str = "."):
        """Downloads a file by its ID.
        If it's a Google Workspace document, it's exported as PDF.
        Otherwise, it's downloaded in its original format.

        Args:
            service: Authenticated Google Drive API service object.
            file_id: The ID of the file to download.
            download_path: The directory path to save the downloaded file. Defaults to current directory.
        """
        GOOGLE_WORKSPACE_MIME_TYPES_TO_EXPORT_AS_PDF = [
            'application/vnd.google-apps.document',
            'application/vnd.google-apps.spreadsheet',
            'application/vnd.google-apps.presentation',
            'application/vnd.google-apps.drawing',
            # Add other Google Workspace types here if needed
        ]

        try:
            # Get file metadata
            file_metadata = service.files().get(fileId=file_id, fields="id, name, mimeType").execute()
            file_name = file_metadata.get('name')
            mime_type = file_metadata.get('mimeType')

            if not file_name:
                print(f"Could not retrieve file name for ID: {file_id}")
                return

            request = None
            output_filename = file_name

            if mime_type in GOOGLE_WORKSPACE_MIME_TYPES_TO_EXPORT_AS_PDF:
                print(f"Exporting Google Workspace file '{file_name}' as PDF...")
                request = service.files().export_media(fileId=file_id, mimeType="application/pdf")
                base_name, _ = os.path.splitext(file_name)
                output_filename = f"{base_name}.pdf"
            else:
                print(f"Downloading file '{file_name}' ({mime_type}) as is...")
                request = service.files().get_media(fileId=file_id)
                # output_filename remains file_name

            if request:
                # Ensure download_path exists
                if not os.path.exists(download_path):
                    os.makedirs(download_path)
                
                file_save_path = os.path.join(download_path, output_filename)
                
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    print(f"Download {int(status.progress() * 100)}%.")
                
                with open(file_save_path, "wb") as f:
                    f.write(fh.getvalue())
                print(f"File '{output_filename}' downloaded successfully to '{file_save_path}'.")

        except HttpError as error:
            print(f"An HttpError occurred: {error}")
            if error.resp.status == 404:
                print(f"File with ID '{file_id}' not found.")
            elif error.resp.status == 403:
                print(f"Permission denied for file ID '{file_id}'. Check scopes or file permissions.")
                print(f"Current scopes are: {SCOPES}. You might need to delete {TOKEN_FILE} and re-authenticate if scopes were changed.")
            else:
                print("Ensure the File ID is correct and you have permissions.")
        except Exception as e:
            print(f"An unexpected error occurred during download: {e}")
        return file_save_path
    if __name__ == '__main__':
        print("Attempting to authenticate with Google Drive...")
        drive_service = authenticate_google_drive()

        if drive_service:
            print("Authentication successful.")
            
            action = input("What would you like to do? (search/download/quit): ").lower()

            if action == "search":
                search_query = input("Enter the search term for Google Drive: ")
                if search_query:
                    print(f"Searching for files containing: '{search_query}'...")
                    found_files = search_drive_files(drive_service, search_query)
                    if found_files:
                        print(f"Found {len(found_files)} file(s):")
                        for f in found_files:
                            print(f"  Name: {f['name']}, ID: {f['id']}, Type: {f['mimeType']}")
                    else:
                        print("No files found matching your query or an error occurred during search.")
                else:
                    print("No search term entered.")
            elif action == "download":
                file_id_to_download = input("Enter the File ID of the file to download: ")
                if file_id_to_download:
                    # You can specify a custom download path here if needed, e.g., "downloads"
                    # download_file_by_id(drive_service, file_id_to_download, "my_downloads")
                    download_file_by_id(drive_service, file_id_to_download)
                else:
                    print("No File ID entered.")
            elif action == "quit":
                print("Exiting.")
            else:
                print("Invalid action. Please choose 'search', 'download', or 'quit'.")

        else:
            print("Failed to authenticate with Google Drive. Please check the console for errors,")
            print(f"ensure '{CLIENT_SECRET_FILE}' is present and correctly configured, and that you have an active internet connection.")
            print("You may need to delete 'token.json' to re-authenticate if issues persist.")
