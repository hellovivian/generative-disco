import {
  Stack,
  Box,
  Typography,
  Button,
  TextField,
  Paper,
} from "@mui/material";
import "../index.css";

function ImagePill({ text }) {
  return (
    <Paper
      id="image_pill"
      sx={{ backgroundColor: "lightpink", width: "150px", height: "180px" }}
    >
      <Typography>{text}</Typography>
      <img
        width="80px"
        height="100px"
        alt="image"
        src="https://cdn-icons-png.flaticon.com/512/61/61122.png"
      />
      <Stack
        direction="row"
        spacing={1}
        sx={{ alignItems: "center", justifyContent: "center" }}
      >
        <Button className="grey_button" id="shuffle">
          <img
            width="20px"
            alt="image"
            src="https://cdn2.iconfinder.com/data/icons/media-player-ui/512/Media-Icon-21-512.png"
          />
        </Button>
        <Button className="grey_button" id="shuffle">
          <img
            width="20px"
            alt="image"
            src="https://cdn-icons-png.flaticon.com/512/3515/3515498.png"
          />
        </Button>
      </Stack>
    </Paper>
  );
}

export default ImagePill;
