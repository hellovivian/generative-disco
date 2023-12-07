import "./App.css";
import Header from "./sections/header";
import Audio from "./sections/audio";
import Video from "./sections/video";
import Tracks from "./sections/tracks";
import { Stack } from "@mui/material";

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <Stack direction="column" spacing={2} width="90%" padding={2}>
          <Header />
          <Stack direction="row" spacing={2} width="100%">
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
