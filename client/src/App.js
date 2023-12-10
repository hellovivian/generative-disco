import "./App.css";
import Header from "./sections/header";
import Audio from "./sections/audio";
import Video from "./sections/video";
import Tracks from "./sections/tracks";
import { Stack, Typography } from "@mui/material";
import React, { useState, useEffect } from "react";

function App() {
  const [data, setData] = useState({ music: "", video: "" });

  useEffect(() => {
    fetch("/just_audio")
      .then((res) => res.json())
      .then((data) => {
        setData(data);
        console.log("Music URL:", data.music);
      })
      .catch((error) => console.error("Error fetching data:", error));
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <Stack direction="column" spacing={2} width="90%" padding={2}>
          <Header />
          <Stack direction="row" spacing={2} width="100%">
            <Typography></Typography>
            <Audio />
            <Video />
          </Stack>
          <Tracks />
        </Stack>
      </header>
    </div>
  );
}

export default App;
