import { Box, Typography, Button } from "@mui/material";
import "../index.css";

function Video() {
  return (
    <Box
      className="container"
      sx={{ width: "25%", borderRadius: 1, padding: 2 }}
    >
      <Typography variant="h5" className="title">
        video area
      </Typography>
      <Button className="grey_button" id="shuffle">
        <img
          width="20px"
          alt="image"
          src="https://cdn-icons-png.flaticon.com/512/3515/3515498.png"
        />
      </Button>
    </Box>
  );
}

export default Video;
