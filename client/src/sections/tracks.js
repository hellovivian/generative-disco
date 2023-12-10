import { Box, Typography, Stack, Button } from "@mui/material";
import "../index.css";
import Track from "../components/track";
import ImagePill from "../components/image-pill";
import Stitch from "../assets/stitch.gif";

function Tracks() {
  return (
    <Box
      className="container"
      sx={{ width: "100%", borderRadius: 1, padding: 2 }}
    >
      <Typography variant="h5" className="title">
        tracks
      </Typography>
      <Stack direction="column" justifyContent="center" alignItems="center">
        <Track />
      </Stack>

      <Button
        className="pink_button"
        sx={{ paddingX: "15px", marginTop: "10px" }}
      >
        <img
          width="60px"
          height="60px"
          alt="stitch"
          src={Stitch}
          style={{ marginRight: "15px" }}
        />
        STICH VIDEO
      </Button>
    </Box>
  );
}

export default Tracks;
