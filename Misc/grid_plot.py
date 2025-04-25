# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Single BT

# %% [markdown]
# ## HTML

# %%
def get_html(n_videos):
    html = """<!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Video Grid</title>
      <link rel="stylesheet" href="styles.css">
    </head>
    <body>
      <div id="grid-container">
        <!-- Example videos, replace with your actual video paths -->
    """
    
    for i in range(n_videos):
        html += f'    <video src="{i}.mp4" class="grid-video" muted loop></video>\n' 
                 
    html += """    <!-- Add more videos -->
      </div>
    
      <!-- Fullscreen Modal -->
      <div id="fullscreen-modal">
        <video id="fullscreen-video" controls></video>
        <div id="video-info">
          <h2 id="video-title"></h2>
          <pre id="video-description"></pre>
        </div>
        <button id="close-btn">Close</button>
      </div>
    
      <script src="script.js"></script>
    </body>
    </html>"""
    return html


# %% [raw]
# print(get_html(5))

# %% [markdown]
# ## CSS

# %%
def get_css(n_columns, n_rows):
    css = """
    /* Grid layout */
    #grid-container {
      display: grid;
    """
    
    css += f"  grid-template-columns: repeat({n_columns}, 1fr);\n"
    css += f"  grid-template-rows: repeat({n_rows}, auto);\n"
    
    css += """
      gap: 10px;
      padding: 10px;
    }
    
    .grid-video {
      width: 89%;
      cursor: pointer;
      border: 2px solid #ccc;
      border-radius: 8px;
      transition: transform 0.3s;
    }
    
    .grid-video:hover {
      transform: scale(1.05);
    }
    
    /* Fullscreen modal */
    #fullscreen-modal {
      display: none;
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(255, 255, 255, 0.9);
      color: black;
      z-index: 1000;
      justify-content: center;
      align-items: center;
      flex-direction: row;
      text-align: center;
      gap: 20px;
      padding: 20px;
    }
    
    #description-container {
      flex: 1;
      display: flex;
      flex-direction: column;
      justify-content: flex-start; /* Align content to the top */
      align-items: flex-start; /* Align the container itself to the left */
      overflow-y: auto;
      padding-left: 20px; /* Optional: Add some padding to avoid content being too close to the edge */
      margin-left: 20px; /* Add space between video and description */
    }
    
    #video-title {
      margin-bottom: 10px;
      font-size: 1.5em;
      font-weight: bold;
    }
    
    #video-description {
      white-space: pre-wrap; /* Preserve spaces and newlines */
      font-size: 1em;
      line-height: 1.5;
      text-align: left; /* Ensure text is left-aligned */
    }
    
    #fullscreen-modal video {
      max-width: 90%;
      max-height: 70%;
    }
    
    #video-info {
      margin-top: 20px;
    }
    
    #close-btn {
      position: absolute;
      bottom: 20%; /* Position button 20px from the bottom */
      left: 50%; /* Center horizontally */
      transform: translateX(-50%); /* Adjust positioning to center the button */
      background-color: #ff4c4c;
      color: white;
      border: none;
      padding: 10px 20px;
      font-size: 1em;
      cursor: pointer;
      border-radius: 5px;
      z-index: 1001; /* Ensure the button stays on top */
    }

    """
    return css


# %% [raw]
# print(get_css(4, 4))

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## JS

# %%
def get_js(videos):
    js = """
    // Get elements
    const gridVideos = document.querySelectorAll('.grid-video');
    const fullscreenModal = document.getElementById('fullscreen-modal');
    const fullscreenVideo = document.getElementById('fullscreen-video');
    const videoTitle = document.getElementById('video-title');
    const videoDescription = document.getElementById('video-description');
    const closeBtn = document.getElementById('close-btn');
    
    // for descriptions
    const videoData = {
    """
    for video in videos:
      js += f"'{video['path']}': " + "{ " +  f"title: '{video['title']}', description: `{video['description']}`" + "},\n"

    js += """};
    
    const coordinates = [
    """
    
    for video in videos:
    
      js += "{" + f"video: '{video['path']}', area: '{video['c_start']} / {video['r_start']} / {video['c_end']} / {video['r_end']}'" + "},\n"
    
    js += """]; 
    // Assign coordinates dynamically
    coordinates.forEach((item, index) => {
      const video = gridVideos[index];
      video.src = item.video; // Set the video source
      video.style.gridArea = item.area; // Set the grid area
    });
    
    // Play all videos on page load
    gridVideos.forEach(video => {
      video.play();
    });
    
    // Open fullscreen on video click
    gridVideos.forEach(video => {
      video.addEventListener('click', () => {
        const src = video.getAttribute('src');
        fullscreenVideo.setAttribute('src', src);
        fullscreenVideo.setAttribute('loop', ''); // Add loop attribute
        fullscreenVideo.setAttribute('autoplay', '');
        fullscreenVideo.play();
    
        // Set video info
        const data = videoData[src];
        videoTitle.textContent = data?.title || 'Untitled';
        videoDescription.textContent = data?.description || 'No description available.';
    
        // Show modal
        fullscreenModal.style.display = 'flex';
      });
    });
    
    // Close fullscreen
    closeBtn.addEventListener('click', () => {
      fullscreenModal.style.display = 'none';
      fullscreenVideo.pause();
      fullscreenVideo.removeAttribute('src'); // Stop the video
      fullscreenVideo.removeAttribute('loop'); // Remove loop attribute
      fullscreenVideo.removeAttribute('autoplay'); // Remove loop attribute
    });
    // Add event listener for closing the modal with the Escape key
    document.addEventListener('keydown', function(event) {
      if (event.key === 'Escape') {
        closeFullscreen();
      }
    });
    
    // Function to close the fullscreen modal
    function closeFullscreen() {
      fullscreenModal.style.display = 'none';
      fullscreenVideo.pause();
      fullscreenVideo.removeAttribute('src'); // Reset video
    }
    """
    return js


# %% [raw]
# videos = [{"path": "banana.mp4", "title": "kiwi", "description": "papaya", "c_start":0, "c_end": 1, "r_start": 2, "r_end": 3}]
# print(get_js(videos))

# %%

# %% [markdown]
# # Bi-BT

# %% [markdown]
# ## HTML

# %%
def get_bi_html(n_videos):
    html = """<!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Video Grid</title>
      <link rel="stylesheet" href="styles.css">
    </head>
    <body>
      <div id="grid-container">
        <!-- Example videos, replace with your actual video paths -->
    """
    
    for i in range(n_videos):
        html += f'    <video src="{i}.mp4" class="grid-video" muted loop></video>\n' 
                 
    html += """    <!-- Add more videos -->
      </div>
    
      <!-- Fullscreen Modal -->
      <div id="fullscreen-modal">
        <div id="left-description-container" class="description-container">
            <h2>Attacker (red)</h2>
            <pre id="left-description"></pre>
        </div>

        <div id="video-info">
          <h2 id="video-title"></h2>
          <video id="fullscreen-video" controls></video>
        </div>
        
        <!-- Right Description -->
        <div id="right-description-container" class="description-container">
            <h2>Defender (blue)</h2>
            <pre id="right-description"></pre>
        </div>
        <button id="close-btn">Close</button>
      </div>
    
      <script src="script.js"></script>
    </body>
    </html>"""
    return html


# %% [markdown]
# ## CSS

# %%
def get_bi_css(n_columns, n_rows):
    css = """
    /* Grid layout */
    #grid-container {
      display: grid;
    """
    
    css += f"  grid-template-columns: repeat({n_columns}, 1fr);\n"
    css += f"  grid-template-rows: repeat({n_rows}, auto);\n"
    
    css += """
      gap: 10px;
      padding: 10px;
    }
    
    .grid-video {
      width: 89%;
      cursor: pointer;
      border: 2px solid #ccc;
      border-radius: 8px;
      transition: transform 0.3s;
    }
    
    .grid-video:hover {
      transform: scale(1.05);
    }
    
    /* Fullscreen modal */
    #fullscreen-modal {
      display: none;
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(255, 255, 255, 0.9);
      color: black;
      z-index: 1000;
      justify-content: center;
      align-items: center;
      flex-direction: row;
      text-align: center;
      gap: 20px;
      padding: 20px;
    }
    
    #description-container {
      flex: 1;
      display: flex;
      flex-direction: column;
      justify-content: flex-start; /* Align content to the top */
      align-items: flex-start; /* Align the container itself to the left */
      overflow-y: auto;
      padding-left: 20px; /* Optional: Add some padding to avoid content being too close to the edge */
      margin-left: 20px; /* Add space between video and description */
    }
    
    #left-description {
      text-align: left;
      white-space: pre-wrap; /* Preserve spaces and newlines */
    }
    
    #right-description {
      text-align: left;
      white-space: pre-wrap; /* Preserve spaces and newlines */
    }

    #video-title {
      margin-bottom: 10px;
      font-size: 1.5em;
      font-weight: bold;
    }
    
    #video-description {
      white-space: pre-wrap; /* Preserve spaces and newlines */
      font-size: 1em;
      line-height: 1.5;
      text-align: left; /* Ensure text is left-aligned */
    }
    
    #fullscreen-modal video {
      max-width: 90%;
      max-height: 70%;
    }
    
    #video-info {
      margin-top: 20px;
    }
    
    #close-btn {
      position: absolute;
      bottom: 20%; /* Position button 20px from the bottom */
      left: 50%; /* Center horizontally */
      transform: translateX(-50%); /* Adjust positioning to center the button */
      background-color: #ff4c4c;
      color: white;
      border: none;
      padding: 10px 20px;
      font-size: 1em;
      cursor: pointer;
      border-radius: 5px;
      z-index: 1001; /* Ensure the button stays on top */
    }

    """
    return css


# %% [markdown]
# ## JS

# %%
def get_bi_js(videos):
    js = """
    // Get elements
    const gridVideos = document.querySelectorAll('.grid-video');
    const fullscreenModal = document.getElementById('fullscreen-modal');
    const fullscreenVideo = document.getElementById('fullscreen-video');
    const videoTitle = document.getElementById('video-title');
    const videoDescription = document.getElementById('video-description');
    const closeBtn = document.getElementById('close-btn');
    
    // for descriptions
    const videoData = {
    """
    for video in videos:
      js += f"'{video['path']}': " + "{ " +  f"title: '{video['title']}', leftDescription: `{video['leftDescription']}`, rightDescription: `{video['rightDescription']}`" + "},\n"

    js += """};
    
    const coordinates = [
    """
    
    for video in videos:
    
      js += "{" + f"video: '{video['path']}', area: '{video['c_start']} / {video['r_start']} / {video['c_end']} / {video['r_end']}'" + "},\n"
    
    js += """]; 
    // Assign coordinates dynamically
    coordinates.forEach((item, index) => {
      const video = gridVideos[index];
      video.src = item.video; // Set the video source
      video.style.gridArea = item.area; // Set the grid area
    });
    
    // Play all videos on page load
    gridVideos.forEach(video => {
      video.play();
    });
    
    // Open fullscreen on video click
    gridVideos.forEach(video => {
      video.addEventListener('click', () => {
        const src = video.getAttribute('src');
        fullscreenVideo.setAttribute('src', src);
        fullscreenVideo.setAttribute('loop', ''); // Add loop attribute
        fullscreenVideo.setAttribute('autoplay', '');
        fullscreenVideo.play();
    
        // Set video info
        const data = videoData[src];
        videoTitle.textContent = data?.title || 'Untitled';
        
        // Set descriptions (left and right)
        document.getElementById("left-description").textContent = data?.leftDescription || 'No left description available.';
        document.getElementById("right-description").textContent = data?.rightDescription || 'No right description available.';
            
        // Show modal
        fullscreenModal.style.display = 'flex';
      });
    });
    
    // Close fullscreen
    closeBtn.addEventListener('click', () => {
      fullscreenModal.style.display = 'none';
      fullscreenVideo.pause();
      fullscreenVideo.removeAttribute('src'); // Stop the video
      fullscreenVideo.removeAttribute('loop'); // Remove loop attribute
      fullscreenVideo.removeAttribute('autoplay'); // Remove loop attribute
    });
    // Add event listener for closing the modal with the Escape key
    document.addEventListener('keydown', function(event) {
      if (event.key === 'Escape') {
        closeFullscreen();
      }
    });
    
    // Function to close the fullscreen modal
    function closeFullscreen() {
      fullscreenModal.style.display = 'none';
      fullscreenVideo.pause();
      fullscreenVideo.removeAttribute('src'); // Reset video
    }
    """
    return js
