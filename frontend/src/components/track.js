import {
  Stack,
  Box,
  Typography,
  Button,
  TextField,
  Grid,
  Paper,
} from "@mui/material";
import "../index.css";

function Track() {
  const items = [];

  for (let i = 0; i < 12; i++) {
    items.push(
      <Grid item key={i} xs={3} md={2} lg={1}>
        <img
          width="100px"
          height="150px"
          alt="image"
          src="https://cdn-icons-png.flaticon.com/512/61/61122.png"
        />
      </Grid>
  );
    }

  return (
    <Paper
      id="track"
      sx={{
        backgroundColor: "lightpink",
        width: "95%",
        height: "auto",
        padding: "20px",
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <Stack direction="row" spacing={3}>
        <Stack direction="column" spacing={2}>
          <Typography className="title">VIDEO INTERVAL</Typography>
          <img
            width="120px"
            height="150px"
            alt="image"
            src="https://cdn-icons-png.flaticon.com/512/61/61122.png"
          />
          <Button className="grey_button" id="shuffle">
            <img
              width="20px"
              alt="image"
              src="https://cdn-icons-png.flaticon.com/512/3515/3515498.png"
            />
          </Button>
        </Stack>
        <Stack direction="column">
          <Typography className="title">FRAMES</Typography>
          <Grid container directon="row" spacing={1} width="100%">
            {
              items
            }
          </Grid>
        </Stack>
      </Stack>
    </Paper>
  );
}

export default Track;
