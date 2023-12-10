import React from "react";
import { TextField, Button, Grid } from "@mui/material";
import "../index.css";
// TODO: get input outline to look prettier

function SetUpInfo() {
  return (
    <Grid container spacing={2} justifyContent="flex-end">
      <Grid item>
        <TextField
          id="openaikey"
          name="openaikey"
          placeholder="OpenAI Key"
          className="input"
          inputProps={{
            className: "placeholder",
          }}
          required
          size="small"
        />
      </Grid>
      <Grid item>
        <TextField
          id="urltomusic"
          name="urltomusic"
          placeholder="URL to Music"
          className="input"
          inputProps={{
            className: "placeholder",
          }}
          required
          size="small"
        />
      </Grid>
      <Grid item>
        <TextField
          id="music_start"
          name="music_start"
          placeholder="Music Start"
          className="input"
          inputProps={{
            className: "placeholder",
          }}
          required
          size="small"
        />
      </Grid>
      <Grid item>
        <TextField
          id="music_length"
          name="music_length"
          placeholder="Music Length"
          className="input"
          inputProps={{
            className: "placeholder",
          }}
          required
          size="small"
        />
      </Grid>
      <Grid item alignItems="center">
        <Button
          id="musicURLUpload"
          // TODO: onClick
          // onClick={}
          // variant="outlined"
          // color="info"
          className="pink_button"
        >
          Change Music
        </Button>
      </Grid>
    </Grid>
  );
}

export default SetUpInfo;
