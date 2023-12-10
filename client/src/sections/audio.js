import {
  Stack,
  Box,
  Typography,
  Button,
  TextField,
  Paper,
} from "@mui/material";
import "../index.css";
import { PlayArrow, Pause } from "@mui/icons-material";
import { useState } from "react";
import ImagePill from "../components/image-pill";

function Audio() {
  const [play, setPlay] = useState(true);
  const handlePlayToggle = () => {
    setPlay(!play);
  };

  return (
    <Box className="container" sx={{ width: "75%", borderRadius: 1 }}>
      <Stack direction="column" spacing={3} padding={2}>
        <Typography variant="h5" className="title">
          audio area
        </Typography>
        <Stack id="buttons" direction="row" spacing={2}>
          <Button
            className="pink_button"
            onClick={handlePlayToggle}
            sx={{ paddingRight: "15px" }}
          >
            {play ? (
              <PlayArrow sx={{ marginRight: "5px" }} />
            ) : (
              <Pause sx={{ marginRight: "5px" }} />
            )}
            {play ? "Play" : "Pause"}
          </Button>
          <Button className="grey_button">
            <img
              width="24px"
              alt="add"
              src="https://static.thenounproject.com/png/2577717-200.png"
              style={{ marginRight: "10px" }}
            />
            Add Interval
          </Button>
          <Button className="grey_button">
            <img
              width="32px"
              alt="trash"
              src="https://cdn-icons-png.flaticon.com/512/3515/3515498.png"
            />
            Delete
          </Button>
          <Button className="grey_button">
            <img
              width="24px"
              alt="save"
              src="https://icons.veryicon.com/png/o/miscellaneous/utility/save-44.png"
              style={{ marginRight: "10px" }}
            />
            Save
          </Button>
        </Stack>
        <Stack
          id="inputs"
          direction="row"
          spacing={2}
          alignContent="center"
          justifyContent="center"
        >
          <Stack
            direction="row"
            spacing={1}
            alignContent="center"
            justifyContent="center"
          >
            <TextField
              id="brainstorm"
              name="brainstorm"
              placeholder="Describe Prompt"
              className="input"
              inputProps={{
                className: "placeholder",
              }}
              required
              sx={{
                minWidth: "120px",
              }}
            />
            <Button
              className="blue_button"
              onClick={handlePlayToggle}
              sx={{ paddingRight: "15px" }}
            >
              <img
                width="24px"
                alt="lightbulb"
                src="https://cdn-icons-png.flaticon.com/128/5038/5038742.png"
                style={{ marginRight: "10px" }}
              />
              Brainstorm
            </Button>
          </Stack>

          <Stack
            direction="row"
            spacing={1}
            alignContent="center"
            justifyContent="center"
          >
            <Typography variant="subtitle1" className="title">
              interval #:
            </Typography>
            <TextField
              id="interval_number"
              name="interval_number"
              placeholder="1"
              className="input"
              inputProps={{
                className: "placeholder",
              }}
              required
              width="10px"
              sx={{
                width: "60px",
              }}
            />
          </Stack>

          <Stack
            direction="row"
            spacing={1}
            alignContent="center"
            justifyContent="center"
          >
            <Typography variant="subtitle1" className="title">
              begin time:
            </Typography>
            <TextField
              id="begin"
              name="begin"
              placeholder="0"
              className="input"
              inputProps={{
                className: "placeholder",
              }}
              required
              sx={{
                width: "60px",
              }}
            />
          </Stack>

          <Stack
            direction="row"
            spacing={1}
            alignContent="center"
            justifyContent="center"
          >
            <Typography variant="subtitle1" className="title">
              end time:
            </Typography>
            <TextField
              id="end"
              name="end"
              placeholder="0.54"
              className="input"
              inputProps={{
                className: "placeholder",
              }}
              required
              sx={{
                width: "60px",
              }}
            />
          </Stack>
        </Stack>
        <Stack
          id="interval_previews"
          direction="row"
          spacing={2}
          alignContent="center"
          justifyContent="center"
        >
          <Stack
            direction="column"
            spacing={1}
            alignContent="center"
            justifyContent="center"
          >
            <TextField
              id="start_prompt"
              name="start_prompt"
              placeholder="Starting Prompt"
              className="input"
              multiline={true}
              inputProps={{
                className: "placeholder",
              }}
              required
              sx={{
                minWidth: "120px",
              }}
            />
            <TextField
              id="start_seed"
              name="start_seed"
              placeholder="Starting Seed"
              className="input"
              inputProps={{
                className: "placeholder",
              }}
              required
              sx={{
                minWidth: "120px",
              }}
            />
          </Stack>
          <Stack
            direction="column"
            spacing={1}
            alignContent="center"
            justifyContent="center"
          >
            <TextField
              id="end_prompt"
              name="end_prompt"
              placeholder="End Prompt"
              className="input"
              multiline={true}
              inputProps={{
                className: "placeholder",
              }}
              required
              sx={{
                minWidth: "120px",
              }}
            />
            <TextField
              id="end_seed"
              name="end_seed"
              placeholder="End Seed"
              className="input"
              inputProps={{
                className: "placeholder",
              }}
              required
              sx={{
                minWidth: "120px",
              }}
            />
          </Stack>
        </Stack>
        <Paper id="brainstorm" sx={{ backgroundColor: "lightgrey" }}>
          <Typography variant="subtitle1" className="title">
            <img
              width="24px"
              alt="lightbulb"
              src="https://cdn-icons-png.flaticon.com/512/3521/3521848.png"
              style={{ marginRight: "10px" }}
            />
            brainstorming area
          </Typography>
          <Stack direction="column" spacing={2}>
            <Stack
              direction="row"
              spacing={1}
              alignContent="center"
              justifyContent="center"
            >
              <TextField
                id="start_prompt"
                name="start_prompt"
                placeholder="Start Prompt"
                className="input"
                multiline={true}
                inputProps={{
                  className: "placeholder",
                }}
                required
                sx={{
                  minWidth: "400px",
                }}
              />
              <TextField
                id="seed"
                name="seed"
                placeholder="Seed"
                className="input"
                inputProps={{
                  className: "placeholder",
                }}
                required
                sx={{
                  minWidth: "120px",
                }}
              />
              <Button
                className="blue_button"
                onClick={handlePlayToggle}
                sx={{ marginLeft: "10px", paddingRight: "15px" }}
              >
                <img
                  width="32px"
                  alt="preview_img"
                  src="https://cdn-icons-png.flaticon.com/128/3971/3971176.png"
                  style={{ marginRight: "10px" }}
                />
                PREVIEW IMG
              </Button>
            </Stack>
            <Typography variant="subtitle1" className="title">
              <img
                width="20px"
                alt="history"
                src="https://cdn-icons-png.flaticon.com/512/61/61122.png"
                style={{
                  marginRight: "10px",
                  justifyContent: "center",
                  alignContent: "center",
                }}
              />
              HISTORY
            </Typography>
            <ImagePill text="example" />
          </Stack>
        </Paper>
      </Stack>
    </Box>
  );
}

export default Audio;
